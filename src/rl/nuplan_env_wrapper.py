"""
NuplanEnvWrapper
================
A gym-style environment wrapper around the nuplan scenario dataset.

Episode lifecycle
-----------------
  obs, info = env.reset(data_batch)   # load one training batch as initial obs
  while not done:
      obs, reward, cost, done, info = env.step(action, model_out)

"State"  = the nuplan feature dict (data) produced by PlutoFeatureBuilder.
"Action" = flat index into the flattened (R×M) candidate trajectory tensor.
"Reward" = weighted combination of progress, comfort, and on-route bonus.
"Cost"   = constraint violations used by the CMDP (collision, speed, accel).

The wrapper does NOT require a live nuplan SimulationRunner.  It operates on
pre-built feature tensors (from the cache) and simulates reward/cost signals
from the trajectory geometry and map features already present in the tensor.

For each batch element the scenario is treated as a single-step episode
(the full 8-second trajectory horizon) which is the standard formulation
for offline/batch RL on the nuplan dataset.  Multi-step receding-horizon
episodes are supported via env.step() being called repeatedly within the
same batch element if the caller provides updated observations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reward / cost weights (can be overridden via constructor kwargs)
# ---------------------------------------------------------------------------
_DEFAULT_REWARD_WEIGHTS = dict(
    progress=1.0,
    on_route=0.5,
    comfort=0.3,
    goal_proximity=0.5,
)

_DEFAULT_COST_WEIGHTS = dict(
    collision=1.0,
    speed_limit=0.5,
    accel_limit=0.3,
    lat_accel=0.3,
)

# Physical limits used when no semantic constraints are available
_SPEED_LIMIT_MS = 15.0      # m/s  (~54 km/h, conservative urban default)
_MAX_ACCEL_MS2 = 4.0        # m/s²
_MAX_LAT_ACCEL_MS2 = 3.0    # m/s²
_MIN_AGENT_DIST_M = 2.5     # metres clearance to nearest agent


@dataclass
class StepInfo:
    reward: float = 0.0
    cost: float = 0.0
    done: bool = False
    reward_components: Dict[str, float] = field(default_factory=dict)
    cost_components: Dict[str, float] = field(default_factory=dict)
    selected_trajectory: Optional[torch.Tensor] = None   # (T, 4)


class NuplanEnvWrapper:
    """
    Stateful environment wrapper over a nuplan feature batch.

    Parameters
    ----------
    future_steps   : number of future trajectory time-steps (default 80 = 8 s at 10 Hz)
    step_dt        : simulation time step in seconds
    history_steps  : number of history frames, including the current frame
    reward_weights : dict overriding _DEFAULT_REWARD_WEIGHTS
    cost_weights   : dict overriding _DEFAULT_COST_WEIGHTS
    device         : torch device
    """

    def __init__(
        self,
        future_steps: int = 80,
        step_dt: float = 0.1,
        history_steps: int = 21,
        reward_weights: Optional[Dict[str, float]] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.future_steps = future_steps
        self.step_dt = step_dt
        self.rw = {**_DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.cw = {**_DEFAULT_COST_WEIGHTS, **(cost_weights or {})}
        self.device = device or torch.device("cpu")

        # Episode state
        self._data: Optional[Dict] = None
        self._history_steps: int = history_steps
        self._done: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, data: Dict) -> Tuple[Dict, Dict]:
        """
        Begin a new episode with the provided feature batch.

        Args:
            data: nuplan feature dict (on any device; will be moved to self.device)

        Returns:
            observation : same dict (the initial state)
            info        : episode metadata
        """
        self._data = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in data.items() if not isinstance(v, dict)}
        # Recursively move nested dicts
        for k, v in data.items():
            if isinstance(v, dict):
                self._data[k] = {
                    kk: (vv.to(self.device) if isinstance(vv, torch.Tensor) else vv)
                    for kk, vv in v.items()
                }

        self._done = False
        info = {"batch_size": self._batch_size()}
        return self._data, info

    def step(
        self,
        action: torch.Tensor,
        model_out: Dict,
        semantic_constraints: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, StepInfo]:
        """
        Apply action (trajectory index) and compute reward/cost.

        Args:
            action              : (bs,) long tensor – flat index into candidate traj
            model_out           : dict from model forward pass containing at minimum
                                  'candidate_trajectories' (bs, R, M, T, >=3) or
                                  'output_trajectory' (bs, T, 3)
            semantic_constraints: optional dict with 'max_speed', 'max_accel',
                                  'max_lat_accel', 'min_distance' tensors (bs,)
                                  produced by CMDPConstraintModule

        Returns:
            next_obs  : feature dict (unchanged for single-step formulation)
            reward    : (bs,) float tensor
            cost      : (bs,) float tensor
            done      : (bs,) bool tensor
            info      : StepInfo with component breakdowns
        """
        if self._data is None:
            raise RuntimeError("Call env.reset() before env.step().")

        bs = self._batch_size()

        # ----------------------------------------------------------------
        # 1. Retrieve selected trajectory
        # ----------------------------------------------------------------
        selected_traj = self._get_selected_trajectory(action, model_out, bs)
        # selected_traj: (bs, T, C) where C >= 3 [x, y, heading_or_cos, (sin)]

        # ----------------------------------------------------------------
        # 2. Compute reward
        # ----------------------------------------------------------------
        reward, reward_components = self._compute_reward(selected_traj)

        # ----------------------------------------------------------------
        # 3. Compute cost (constraint violations)
        # ----------------------------------------------------------------
        cost, cost_components = self._compute_cost(
            selected_traj, semantic_constraints
        )

        # ----------------------------------------------------------------
        # 4. Episode termination
        # ----------------------------------------------------------------
        # Single-step formulation: episode ends after each trajectory horizon
        self._done = True
        done = torch.ones(bs, dtype=torch.bool, device=self.device)

        info = StepInfo(
            reward=reward.mean().item(),
            cost=cost.mean().item(),
            done=True,
            reward_components=reward_components,
            cost_components=cost_components,
            selected_trajectory=selected_traj,
        )

        return self._data, reward, cost, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_size(self) -> int:
        if self._data is None:
            return 0
        if "agent" in self._data and "position" in self._data["agent"]:
            return self._data["agent"]["position"].shape[0]
        return 1

    def _get_selected_trajectory(
        self, action: torch.Tensor, model_out: Dict, bs: int
    ) -> torch.Tensor:
        """
        Extract the trajectory tensor for each batch element's chosen action.
        Returns (bs, T, C) where C may be 3 (x,y,h) or 4 (x,y,cos,sin).
        """
        if "candidate_trajectories" in model_out:
            cands = model_out["candidate_trajectories"]  # (bs, R, M, T, C)
            if cands.dim() == 5:
                R, M = cands.shape[1], cands.shape[2]
                flat = cands.reshape(bs, R * M, cands.shape[3], cands.shape[4])
                # action: (bs,), clamp to valid range
                idx = action.clamp(0, flat.shape[1] - 1)
                selected = flat[torch.arange(bs, device=cands.device), idx]
            elif cands.dim() == 4:
                # (bs, N, T, C)
                idx = action.clamp(0, cands.shape[1] - 1)
                selected = cands[torch.arange(bs, device=cands.device), idx]
            else:
                selected = cands[:, 0] if cands.dim() == 3 else cands
        elif "output_trajectory" in model_out:
            selected = model_out["output_trajectory"]
        elif "moe_output_trajectory" in model_out:
            selected = model_out["moe_output_trajectory"]
        else:
            # Fallback: zero trajectory
            selected = torch.zeros(bs, self.future_steps, 3, device=self.device)
        return selected

    def _traj_positions(self, traj: torch.Tensor) -> torch.Tensor:
        """Return (bs, T, 2) position slice."""
        return traj[..., :2]

    def _traj_speed(self, traj: torch.Tensor) -> torch.Tensor:
        """Approximate speed (bs, T) from finite differences of position."""
        pos = self._traj_positions(traj)  # (bs, T, 2)
        delta = torch.diff(pos, dim=1, prepend=pos[:, :1])
        speed = torch.norm(delta, dim=-1) / self.step_dt
        return speed

    def _traj_accel(self, speed: torch.Tensor) -> torch.Tensor:
        """Longitudinal acceleration (bs, T) from finite differences of speed."""
        return torch.diff(speed, dim=1, prepend=speed[:, :1]) / self.step_dt

    def _traj_lat_accel(self, traj: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Approximate lateral acceleration from curvature × v².
        Works with both (x,y,h) and (x,y,cos,sin) trajectory formats.
        """
        pos = self._traj_positions(traj)
        v = torch.diff(pos, dim=1, prepend=pos[:, :1])
        # Angle of velocity vector
        angle = torch.atan2(v[..., 1], v[..., 0])
        yaw_rate = torch.diff(angle, dim=1, prepend=angle[:, :1]) / self.step_dt
        lat_accel = torch.abs(yaw_rate * speed.clamp(min=0.1))
        return lat_accel

    def _compute_reward(
        self, traj: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bs = traj.shape[0]
        device = traj.device

        # --- Progress: distance travelled along trajectory ---
        pos = self._traj_positions(traj)
        progress = torch.norm(pos[:, -1] - pos[:, 0], dim=-1)   # (bs,)

        # --- On-route: fraction of trajectory points near reference line ---
        on_route_bonus = torch.zeros(bs, device=device)
        if (
            self._data is not None
            and "reference_line" in self._data
            and "position" in self._data["reference_line"]
        ):
            rl_pos = self._data["reference_line"]["position"]  # (bs, R, Lr, 2 or 3)
            rl_valid = self._data["reference_line"]["valid_mask"]  # (bs, R, Lr)
            if rl_pos.shape[-1] >= 2 and rl_pos.dim() == 4:
                # Use only valid reference line points
                rl_xy = rl_pos[..., :2]   # (bs, R, Lr, 2)
                ego_xy = pos.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, T, 2)
                # Compute min distance to any ref line point per timestep
                rl_exp = rl_xy.unsqueeze(-2)  # (bs, R, Lr, 1, 2)
                ego_exp = pos.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, T, 2)
                dist = torch.norm(rl_exp - ego_exp, dim=-1)  # (bs, R, Lr, T)
                rl_valid_exp = rl_valid.unsqueeze(-1).expand_as(dist)
                dist = dist.masked_fill(~rl_valid_exp, 1e4)
                min_dist = dist.reshape(bs, -1, pos.shape[1]).min(dim=1)[0]  # (bs, T)
                on_route_bonus = (min_dist < 3.0).float().mean(dim=-1)

        # --- Comfort: penalise harsh acceleration ---
        speed = self._traj_speed(traj)
        accel = self._traj_accel(speed)
        comfort = -torch.mean(accel.abs(), dim=-1)  # (bs,) negative = penalty

        # --- Goal proximity ---
        goal_proximity = torch.zeros(bs, device=device)
        if (
            self._data is not None
            and "agent" in self._data
            and "target" in self._data["agent"]
        ):
            target_pos = self._data["agent"]["target"][:, 0, -1, :2]  # (bs, 2)
            final_pos = pos[:, -1]
            dist_to_goal = torch.norm(final_pos - target_pos, dim=-1)
            goal_proximity = torch.exp(-dist_to_goal / 10.0)

        reward = (
            self.rw["progress"] * progress
            + self.rw["on_route"] * on_route_bonus
            + self.rw["comfort"] * comfort
            + self.rw["goal_proximity"] * goal_proximity
        )

        components = {
            "progress": progress.mean().item(),
            "on_route": on_route_bonus.mean().item(),
            "comfort": comfort.mean().item(),
            "goal_proximity": goal_proximity.mean().item(),
        }
        return reward, components

    def _compute_cost(
        self,
        traj: torch.Tensor,
        semantic_constraints: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bs = traj.shape[0]
        device = traj.device

        # Extract per-batch constraint thresholds
        max_speed = (
            semantic_constraints["max_speed"]
            if semantic_constraints is not None and "max_speed" in semantic_constraints
            else torch.full((bs,), _SPEED_LIMIT_MS, device=device)
        )
        max_accel = (
            semantic_constraints["max_accel"]
            if semantic_constraints is not None and "max_accel" in semantic_constraints
            else torch.full((bs,), _MAX_ACCEL_MS2, device=device)
        )
        max_lat_accel = (
            semantic_constraints["max_lat_accel"]
            if semantic_constraints is not None and "max_lat_accel" in semantic_constraints
            else torch.full((bs,), _MAX_LAT_ACCEL_MS2, device=device)
        )
        min_dist = (
            semantic_constraints["min_distance"]
            if semantic_constraints is not None and "min_distance" in semantic_constraints
            else torch.full((bs,), _MIN_AGENT_DIST_M, device=device)
        )

        speed = self._traj_speed(traj)          # (bs, T)
        accel = self._traj_accel(speed)         # (bs, T)
        lat_a = self._traj_lat_accel(traj, speed)  # (bs, T)

        # Speed violation
        c_speed = F.relu(speed - max_speed.unsqueeze(-1)).mean(dim=-1)

        # Acceleration violation
        c_accel = F.relu(accel.abs() - max_accel.unsqueeze(-1)).mean(dim=-1)

        # Lateral acceleration violation
        c_lat = F.relu(lat_a - max_lat_accel.unsqueeze(-1)).mean(dim=-1)

        # Collision / proximity violation
        c_coll = torch.zeros(bs, device=device)
        if (
            self._data is not None
            and "agent" in self._data
            and "position" in self._data["agent"]
        ):
            agent_pos = self._data["agent"]["position"]  # (bs, A, history+future, 2)
            agent_valid = self._data["agent"]["valid_mask"]  # (bs, A, T)
            A = agent_pos.shape[1]
            if A > 1:
                # Use last history position as proxy for current agent position
                hist_end = self._history_steps - 1 if self._history_steps > 0 else -1
                other_pos = agent_pos[:, 1:, hist_end, :2]     # (bs, A-1, 2)
                other_valid = agent_valid[:, 1:, hist_end]      # (bs, A-1)
                ego_pos = self._traj_positions(traj)            # (bs, T, 2)

                # Min distance over time and agents
                dist = torch.norm(
                    other_pos.unsqueeze(2) - ego_pos.unsqueeze(1), dim=-1
                )  # (bs, A-1, T)
                mask = other_valid.unsqueeze(-1).expand_as(dist)
                dist = dist.masked_fill(~mask, 1e4)
                min_d = dist.min(dim=1)[0]  # (bs, T)
                c_coll = F.relu(min_dist.unsqueeze(-1) - min_d).mean(dim=-1)

        cost = (
            self.cw["collision"] * c_coll
            + self.cw["speed_limit"] * c_speed
            + self.cw["accel_limit"] * c_accel
            + self.cw["lat_accel"] * c_lat
        )
        components = {
            "collision": c_coll.mean().item(),
            "speed_limit": c_speed.mean().item(),
            "accel_limit": c_accel.mean().item(),
            "lat_accel": c_lat.mean().item(),
        }
        return cost, components
