"""
RolloutBuffer
=============
On-policy rollout storage for PPO with CMDP support.

Stores per-step tensors:
  observations : list of feature dicts (nuplan data batches)
  actions      : (N, bs) long   – flat candidate indices
  log_probs    : (N, bs) float  – log π_old(a|s)
  rewards      : (N, bs) float  – environment reward signal
  costs        : (N, bs) float  – CMDP constraint cost signal
  values       : (N, bs) float  – V(s) from critic
  cost_values  : (N, bs) float  – Vc(s) from cost-critic
  dones        : (N, bs) bool   – episode termination flag

After ``finish_rollout()`` is called the buffer computes:
  advantages       : (N, bs) – GAE(γ, λ) over reward
  cost_advantages  : (N, bs) – GAE(γ_c, λ_c) over cost
  returns          : (N, bs) – reward-to-go targets for value head
  cost_returns     : (N, bs) – cost-to-go targets for cost-value head
"""

from typing import Dict, List, Optional

import torch


class RolloutBuffer:
    """
    Fixed-capacity rollout buffer.

    Parameters
    ----------
    capacity    : maximum number of rollout steps to store
    gamma       : discount factor for reward
    gamma_cost  : discount factor for cost
    lam         : GAE lambda for reward advantage
    lam_cost    : GAE lambda for cost advantage
    device      : where tensors are stored
    """

    def __init__(
        self,
        capacity: int = 256,
        gamma: float = 0.99,
        gamma_cost: float = 0.99,
        lam: float = 0.95,
        lam_cost: float = 0.95,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = capacity
        self.gamma = gamma
        self.gamma_cost = gamma_cost
        self.lam = lam
        self.lam_cost = lam_cost
        self.device = device or torch.device("cpu")

        self._ptr = 0
        self._full = False

        # Scalar-tensor storage (lazily initialised on first add())
        self._actions: Optional[torch.Tensor] = None
        self._log_probs: Optional[torch.Tensor] = None
        self._rewards: Optional[torch.Tensor] = None
        self._costs: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._cost_values: Optional[torch.Tensor] = None
        self._dones: Optional[torch.Tensor] = None

        # Observation list (nuplan dicts cannot be pre-allocated as tensors)
        self._observations: List[Dict] = []

        # Computed by finish_rollout()
        self._advantages: Optional[torch.Tensor] = None
        self._cost_advantages: Optional[torch.Tensor] = None
        self._returns: Optional[torch.Tensor] = None
        self._cost_returns: Optional[torch.Tensor] = None

        self._bs: Optional[int] = None

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer and reset the pointer."""
        self._ptr = 0
        self._full = False
        self._actions = None
        self._log_probs = None
        self._rewards = None
        self._costs = None
        self._values = None
        self._cost_values = None
        self._dones = None
        self._observations = []
        self._advantages = None
        self._cost_advantages = None
        self._returns = None
        self._cost_returns = None
        self._bs = None

    def add(
        self,
        obs: Dict,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        value: torch.Tensor,
        cost_value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Store one rollout step.

        All tensor arguments should be 1-D with shape (bs,).
        obs is a nuplan feature dict.
        """
        bs = action.shape[0]

        if self._ptr >= self.capacity:
            raise RuntimeError(
                "RolloutBuffer is full. Call finish_rollout() then reset() first."
            )

        # Lazy allocation
        if self._actions is None:
            self._bs = bs
            self._actions = torch.zeros(self.capacity, bs, dtype=torch.long, device=self.device)
            self._log_probs = torch.zeros(self.capacity, bs, device=self.device)
            self._rewards = torch.zeros(self.capacity, bs, device=self.device)
            self._costs = torch.zeros(self.capacity, bs, device=self.device)
            self._values = torch.zeros(self.capacity, bs, device=self.device)
            self._cost_values = torch.zeros(self.capacity, bs, device=self.device)
            self._dones = torch.zeros(self.capacity, bs, dtype=torch.bool, device=self.device)
            self._observations = [None] * self.capacity

        self._observations[self._ptr] = obs
        self._actions[self._ptr] = action.to(self.device)
        self._log_probs[self._ptr] = log_prob.to(self.device).detach()
        self._rewards[self._ptr] = reward.to(self.device).detach()
        self._costs[self._ptr] = cost.to(self.device).detach()
        self._values[self._ptr] = value.to(self.device).detach()
        self._cost_values[self._ptr] = cost_value.to(self.device).detach()
        self._dones[self._ptr] = done.to(self.device)
        self._ptr += 1

    def finish_rollout(
        self,
        last_value: torch.Tensor,
        last_cost_value: torch.Tensor,
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        Call this once at the end of a rollout before iterating over the data.

        Args:
            last_value      : (bs,) – V(s_{T+1}) bootstrap value (0 if terminal)
            last_cost_value : (bs,) – Vc(s_{T+1}) bootstrap cost value
        """
        N = self._ptr
        bs = self._bs

        self._advantages = torch.zeros(N, bs, device=self.device)
        self._cost_advantages = torch.zeros(N, bs, device=self.device)

        last_gae = torch.zeros(bs, device=self.device)
        last_cost_gae = torch.zeros(bs, device=self.device)

        next_val = last_value.to(self.device).detach()
        next_cost_val = last_cost_value.to(self.device).detach()

        for t in reversed(range(N)):
            not_done = (~self._dones[t]).float()
            delta = (
                self._rewards[t]
                + self.gamma * next_val * not_done
                - self._values[t]
            )
            cost_delta = (
                self._costs[t]
                + self.gamma_cost * next_cost_val * not_done
                - self._cost_values[t]
            )

            last_gae = delta + self.gamma * self.lam * not_done * last_gae
            last_cost_gae = cost_delta + self.gamma_cost * self.lam_cost * not_done * last_cost_gae

            self._advantages[t] = last_gae
            self._cost_advantages[t] = last_cost_gae

            next_val = self._values[t]
            next_cost_val = self._cost_values[t]

        self._returns = self._advantages + self._values[:N]
        self._cost_returns = self._cost_advantages + self._cost_values[:N]

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._ptr

    @property
    def is_ready(self) -> bool:
        return self._advantages is not None

    def get_statistics(self) -> Dict[str, float]:
        """Summary statistics for logging."""
        N = self._ptr
        return {
            "mean_reward": self._rewards[:N].mean().item(),
            "mean_cost": self._costs[:N].mean().item(),
            "mean_value": self._values[:N].mean().item(),
            "mean_advantage": self._advantages[:N].mean().item() if self._advantages is not None else 0.0,
            "mean_cost_advantage": self._cost_advantages[:N].mean().item() if self._cost_advantages is not None else 0.0,
        }

    def iterate_minibatches(self, minibatch_size: int, shuffle: bool = True):
        """
        Yield (obs_batch, action_batch, log_prob_batch, advantage_batch,
               return_batch, cost_advantage_batch, cost_return_batch) tuples.

        Observations are yielded as lists of dicts (not stacked) because nuplan
        feature dicts have variable-length tensors.

        Advantages are normalised (mean 0, std 1) per minibatch for training
        stability.
        """
        if not self.is_ready:
            raise RuntimeError("Call finish_rollout() before iterating minibatches.")

        N = self._ptr
        flat_N = N * self._bs

        # Flatten N×bs → flat_N
        actions = self._actions[:N].reshape(flat_N)
        log_probs = self._log_probs[:N].reshape(flat_N)
        advantages = self._advantages[:N].reshape(flat_N)
        returns = self._returns[:N].reshape(flat_N)
        cost_advantages = self._cost_advantages[:N].reshape(flat_N)
        cost_returns = self._cost_returns[:N].reshape(flat_N)

        # Step indices (to recover the matching observation)
        step_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(N, self._bs).reshape(flat_N)
        batch_idx = torch.arange(self._bs, device=self.device).unsqueeze(0).expand(N, self._bs).reshape(flat_N)

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std

        c_adv_mean = cost_advantages.mean()
        c_adv_std = cost_advantages.std(unbiased=False).clamp(min=1e-8)
        cost_advantages = (cost_advantages - c_adv_mean) / c_adv_std

        if shuffle:
            perm = torch.randperm(flat_N, device=self.device)
            actions = actions[perm]
            log_probs = log_probs[perm]
            advantages = advantages[perm]
            returns = returns[perm]
            cost_advantages = cost_advantages[perm]
            cost_returns = cost_returns[perm]
            step_idx = step_idx[perm]
            batch_idx = batch_idx[perm]

        start = 0
        while start < flat_N:
            end = min(start + minibatch_size, flat_N)
            sl = slice(start, end)

            # Build observation sub-batch: collect the unique step indices,
            # then sub-index by within-step batch indices.
            # For simplicity we yield the raw step+batch indices and the
            # caller can reconstruct if needed.
            yield {
                "step_idx": step_idx[sl],
                "batch_idx": batch_idx[sl],
                "actions": actions[sl],
                "log_probs_old": log_probs[sl],
                "advantages": advantages[sl],
                "returns": returns[sl],
                "cost_advantages": cost_advantages[sl],
                "cost_returns": cost_returns[sl],
            }
            start = end

    def get_obs_at(self, step: int) -> Optional[Dict]:
        """Return the stored observation dict at the given step index."""
        if step < 0 or step >= self._ptr:
            return None
        return self._observations[step]
