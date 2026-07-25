"""
CMDPDualUpdater
===============
Constrained MDP (CMDP) dual-variable update via the Lagrangian method.

Theory
------
The CMDP objective (Altman 1999, Achiam et al. 2017):

  max_π  E[∑ γ^t r_t]
  s.t.   E[∑ γ^t c_i_t] ≤ d_i   for i = 1 … K

The Lagrangian relaxation converts this to:

  L(π, λ) = E[∑ γ^t r_t] - ∑_i λ_i (E[∑ γ^t c_i_t] - d_i)

Dual update (gradient ascent on λ):
  λ_i ← max(0,  λ_i + α_λ · (J_c_i − d_i))

where J_c_i is the empirical mean cost for constraint i.

Shielding / projection
----------------------
``shield_trajectory`` projects a candidate trajectory onto the feasible set
defined by the *semantic* constraints from CMDPConstraintModule.  Any
timestep that violates a hard bound is clipped rather than discarded, so the
output trajectory always has the same length.

Supported constraint types
---------------------------
  speed     : ‖v_t‖ ≤ c_max_speed
  accel     : |a_t| ≤ c_max_accel
  distance  : dist(ego, nearest_agent) ≥ c_min_dist  [soft projection only]
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ConstraintSpec:
    """Specification for a single CMDP inequality constraint."""
    name: str
    threshold: float          # cost budget d_i (upper bound on expected cost)
    init_lambda: float = 0.1  # initial dual variable value
    lambda_lr: float = 0.01   # dual update step size
    lambda_max: float = 10.0  # clip λ to avoid runaway


_CONSTRAINT_NAME_BY_CONFIG_KEY = {
    "collision_threshold": "collision",
    "speed_limit_threshold": "speed_limit",
    "accel_limit_threshold": "accel_limit",
    "lat_accel_threshold": "lat_accel",
}

_CONSTRAINT_DUAL_DEFAULTS = {
    "collision": (0.5, 0.02),
    "speed_limit": (0.2, 0.01),
    "accel_limit": (0.1, 0.01),
    "lat_accel": (0.1, 0.01),
}


def build_constraint_specs(
    constraints_cfg: Optional[Dict[str, float]],
) -> Optional[List[ConstraintSpec]]:
    """Convert rl.constraints thresholds into CMDP constraint specifications."""
    if not constraints_cfg:
        return None

    specs = []
    for config_key, name in _CONSTRAINT_NAME_BY_CONFIG_KEY.items():
        if config_key not in constraints_cfg:
            continue
        init_lambda, lambda_lr = _CONSTRAINT_DUAL_DEFAULTS[name]
        specs.append(
            ConstraintSpec(
                name=name,
                threshold=float(constraints_cfg[config_key]),
                init_lambda=init_lambda,
                lambda_lr=lambda_lr,
            )
        )
    return specs or None


class CMDPDualUpdater:
    """
    Maintains and updates Lagrange multipliers λ for K safety constraints.

    Parameters
    ----------
    constraint_specs : list of ConstraintSpec (one per constraint type)
    device           : torch device for λ tensors
    """

    def __init__(
        self,
        constraint_specs: Optional[List[ConstraintSpec]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")

        if constraint_specs is None:
            constraint_specs = [
                ConstraintSpec("collision",   threshold=0.05, init_lambda=0.5,  lambda_lr=0.02),
                ConstraintSpec("speed_limit", threshold=0.10, init_lambda=0.2,  lambda_lr=0.01),
                ConstraintSpec("accel_limit", threshold=0.15, init_lambda=0.1,  lambda_lr=0.01),
                ConstraintSpec("lat_accel",   threshold=0.15, init_lambda=0.1,  lambda_lr=0.01),
            ]

        self.specs = constraint_specs
        self.K = len(self.specs)

        # λ tensors – one scalar per constraint
        self.lambdas: List[torch.Tensor] = [
            torch.tensor(s.init_lambda, dtype=torch.float32, device=self.device)
            for s in self.specs
        ]

    # ------------------------------------------------------------------
    # Dual update
    # ------------------------------------------------------------------

    def update(self, mean_costs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform one dual gradient-ascent step.

        Args:
            mean_costs: dict mapping constraint name → mean cost over last rollout
                        (e.g. from NuplanEnvWrapper StepInfo.cost_components)

        Returns:
            dict mapping constraint name → updated λ value
        """
        updated = {}
        for i, spec in enumerate(self.specs):
            j_c = mean_costs.get(spec.name, 0.0)
            grad = j_c - spec.threshold
            new_lambda = self.lambdas[i] + spec.lambda_lr * grad
            new_lambda = new_lambda.clamp(min=0.0, max=spec.lambda_max)
            self.lambdas[i] = new_lambda
            updated[spec.name] = new_lambda.item()
            logger.debug(
                "CMDP λ[%s]: J_c=%.4f  d=%.4f  grad=%.4f  λ=%.4f",
                spec.name, j_c, spec.threshold, grad, new_lambda.item(),
            )
        return updated

    def get_lambda_tensor(self) -> torch.Tensor:
        """Return all λ as a (K,) tensor for use in PPOUpdater."""
        return torch.stack(self.lambdas).to(self.device)

    def get_total_lambda(self) -> float:
        """Scalar sum of all λ values (used as the single penalty in PPO)."""
        return sum(l.item() for l in self.lambdas)

    def state_dict(self) -> Dict:
        return {spec.name: self.lambdas[i].item() for i, spec in enumerate(self.specs)}

    def load_state_dict(self, d: Dict) -> None:
        for i, spec in enumerate(self.specs):
            if spec.name in d:
                self.lambdas[i] = torch.tensor(d[spec.name], device=self.device)

    # ------------------------------------------------------------------
    # Trajectory shielding / projection
    # ------------------------------------------------------------------

    @staticmethod
    def shield_trajectory(
        trajectory: torch.Tensor,
        semantic_constraints: Dict[str, torch.Tensor],
        step_dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Project a candidate trajectory onto the feasible set.

        Hard constraints applied per timestep (conservative, causal):
          - Speed: if |v_t| > max_speed, rescale displacement proportionally.
          - Acceleration: if |a_t| > max_accel, clip the velocity change.

        Proximity to other agents is a soft constraint (cannot be guaranteed
        by modifying the ego trajectory alone) so it is not projected here.

        Args:
            trajectory           : (bs, T, C)  C ≥ 2  [x, y, ...]
            semantic_constraints : dict with 'max_speed' (bs,), 'max_accel' (bs,)
            step_dt              : time step in seconds

        Returns:
            shielded trajectory (bs, T, C) – same shape as input
        """
        traj = trajectory.clone()
        bs, T, C = traj.shape
        device = traj.device

        max_speed = semantic_constraints.get(
            "max_speed",
            torch.full((bs,), 15.0, device=device)
        ).to(device)  # (bs,)

        max_accel = semantic_constraints.get(
            "max_accel",
            torch.full((bs,), 4.0, device=device)
        ).to(device)  # (bs,)

        # Iterative causal projection
        for t in range(1, T):
            prev_pos = traj[:, t - 1, :2]
            curr_pos = traj[:, t, :2]
            disp = curr_pos - prev_pos
            dist = torch.norm(disp, dim=-1, keepdim=True).clamp(min=1e-6)  # (bs,1)
            speed = dist.squeeze(-1) / step_dt  # (bs,)

            # --- Speed projection ---
            exceed_speed = speed > max_speed
            if exceed_speed.any():
                scale = (max_speed / speed.clamp(min=1e-6)).unsqueeze(-1)  # (bs,1)
                scale = torch.where(
                    exceed_speed.unsqueeze(-1),
                    scale,
                    torch.ones_like(scale),
                )
                new_disp = disp * scale
                traj[:, t, :2] = prev_pos + new_disp

            if t >= 2:
                prev_speed = torch.norm(traj[:, t - 1, :2] - traj[:, t - 2, :2], dim=-1) / step_dt
                curr_speed_after = torch.norm(traj[:, t, :2] - traj[:, t - 1, :2], dim=-1) / step_dt
                accel = (curr_speed_after - prev_speed) / step_dt

                exceed_accel = accel.abs() > max_accel
                if exceed_accel.any():
                    clamped_accel = accel.clamp(-max_accel, max_accel)
                    target_speed = (prev_speed + clamped_accel * step_dt).clamp(min=0.0)
                    curr_disp = traj[:, t, :2] - traj[:, t - 1, :2]
                    curr_dist = torch.norm(curr_disp, dim=-1, keepdim=True).clamp(min=1e-6)
                    direction = curr_disp / curr_dist
                    new_dist = (target_speed * step_dt).unsqueeze(-1)
                    traj[:, t, :2] = torch.where(
                        exceed_accel.unsqueeze(-1),
                        traj[:, t - 1, :2] + direction * new_dist,
                        traj[:, t, :2],
                    )

        return traj

    # ------------------------------------------------------------------
    # Safety monitor (inference-time)
    # ------------------------------------------------------------------

    def is_safe(
        self,
        cost_components: Dict[str, float],
        slack: float = 0.0,
    ) -> bool:
        """
        Return True iff all constraint costs are within their thresholds + slack.
        Useful as a runtime safety check before committing to a trajectory.
        """
        for spec in self.specs:
            val = cost_components.get(spec.name, 0.0)
            if val > spec.threshold + slack:
                return False
        return True
