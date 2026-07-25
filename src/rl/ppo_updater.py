"""
PPOUpdater
==========
Proximal Policy Optimisation update for the PLUTO actor-critic.

Implements the clipped surrogate objective (Schulman et al. 2017) extended
for CMDP: the Lagrange multipliers λ from CMDPDualUpdater are folded into
the policy loss to constrain cost.

Loss breakdown
--------------
  L_policy  = -E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
             - λ · E[min(r_t·A_c_t, clip(r_t, 1-ε, 1+ε)·A_c_t)]
  L_value   = 0.5 · MSE(V(s), R_t)
  L_cost_v  = 0.5 · MSE(Vc(s), Rc_t)
  L_entropy = -H[π(·|s)]
  L_total   = L_policy + c_v·L_value + c_cv·L_cost_v - c_e·L_entropy

Reference
---------
Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
Achiam et al., "Constrained Policy Optimization", 2017 (CMDP extension).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pluto.modules.actor_critic import ActorCriticWrapper

logger = logging.getLogger(__name__)


class PPOUpdater:
    """
    Performs K epochs of PPO mini-batch updates on the actor-critic model.

    Parameters
    ----------
    model          : ActorCriticWrapper wrapping the PlanningModel
    optimizer      : pre-built optimizer (shared with IL training is fine)
    clip_eps       : PPO clipping epsilon (default 0.2)
    value_coef     : coefficient for value loss (default 0.5)
    cost_value_coef: coefficient for cost-value loss (default 0.5)
    entropy_coef   : entropy bonus coefficient (default 0.01)
    max_grad_norm  : gradient clipping (default 0.5)
    ppo_epochs     : number of PPO epochs per update call (default 4)
    target_kl      : early-stop KL divergence threshold (default 0.01)
    """

    def __init__(
        self,
        model: ActorCriticWrapper,
        optimizer: torch.optim.Optimizer,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        cost_value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        target_kl: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.cost_value_coef = cost_value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        buffer,                        # RolloutBuffer
        lagrange_multipliers: torch.Tensor,  # (num_constraints,)
        minibatch_size: int = 64,
        il_loss_fn=None,               # optional callable(data) -> IL loss tensor
        il_loss_weight: float = 0.1,
    ) -> Dict[str, float]:
        """
        Run PPO_epochs of mini-batch updates.

        Args:
            buffer              : filled RolloutBuffer (finish_rollout() called)
            lagrange_multipliers: λ from CMDPDualUpdater, shape (num_constraints,)
            minibatch_size      : elements per mini-batch
            il_loss_fn          : optional IL/imitation loss function for
                                  teacher-regularised RL (DAgger-style)
            il_loss_weight      : weight of IL loss in total loss

        Returns:
            dict of mean losses across all mini-batches
        """
        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "cost_value_loss": [],
            "entropy_loss": [], "total_loss": [], "kl_divergence": [],
            "il_loss": [],
        }

        λ = lagrange_multipliers.sum().item()   # scalar Lagrange penalty

        early_stop = False
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break

            for mb in buffer.iterate_minibatches(minibatch_size, shuffle=True):
                step_indices = mb["step_idx"]         # (mb_size,)
                batch_indices = mb["batch_idx"]       # (mb_size,)
                actions = mb["actions"]               # (mb_size,)
                log_probs_old = mb["log_probs_old"]   # (mb_size,)
                advantages = mb["advantages"]         # (mb_size,)
                returns = mb["returns"]               # (mb_size,)
                cost_advantages = mb["cost_advantages"]
                cost_returns = mb["cost_returns"]

                # Collect unique step indices and rebuild observation batches
                unique_steps = step_indices.unique()
                loss_list = []

                for s in unique_steps:
                    mask = step_indices == s
                    b_idx = batch_indices[mask]

                    obs = buffer.get_obs_at(s.item())
                    if obs is None:
                        continue

                    sub_obs = self._index_obs(obs, b_idx)
                    sub_actions = actions[mask]
                    sub_lp_old = log_probs_old[mask]
                    sub_adv = advantages[mask]
                    sub_ret = returns[mask]
                    sub_cost_adv = cost_advantages[mask]
                    sub_cost_ret = cost_returns[mask]

                    # Forward pass under current policy
                    log_prob, entropy, value, cost_value = self.model.evaluate_actions(
                        sub_obs, sub_actions
                    )

                    # KL divergence (early stop criterion)
                    with torch.no_grad():
                        kl = (sub_lp_old - log_prob).mean()

                    if epoch > 0 and kl.item() > self.target_kl:
                        logger.debug(
                            "Early stopping PPO at epoch %d, KL=%.4f", epoch, kl.item()
                        )
                        early_stop = True
                        break

                    # Policy (actor) loss with CMDP penalty
                    ratio = torch.exp(log_prob - sub_lp_old)
                    clip_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)

                    surr_r = torch.min(ratio * sub_adv, clip_ratio * sub_adv)
                    surr_c = torch.min(ratio * sub_cost_adv, clip_ratio * sub_cost_adv)

                    policy_loss = -surr_r.mean() + λ * surr_c.mean()

                    # Value losses (clipped)
                    v_loss = 0.5 * F.mse_loss(value, sub_ret.detach())
                    cv_loss = 0.5 * F.mse_loss(cost_value, sub_cost_ret.detach())

                    # Entropy bonus
                    ent_loss = -entropy.mean()

                    # Optional IL regularisation
                    il_loss = torch.tensor(0.0, device=value.device)
                    if il_loss_fn is not None:
                        il_loss = il_loss_fn(sub_obs)

                    total = (
                        policy_loss
                        + self.value_coef * v_loss
                        + self.cost_value_coef * cv_loss
                        + self.entropy_coef * ent_loss
                        + il_loss_weight * il_loss
                    )
                    loss_list.append(total)

                    metrics["policy_loss"].append(policy_loss.item())
                    metrics["value_loss"].append(v_loss.item())
                    metrics["cost_value_loss"].append(cv_loss.item())
                    metrics["entropy_loss"].append(ent_loss.item())
                    metrics["total_loss"].append(total.item())
                    metrics["kl_divergence"].append(kl.item())
                    metrics["il_loss"].append(il_loss.item())

                if early_stop:
                    break

                if loss_list:
                    batch_loss = torch.stack(loss_list).mean()
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

        return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_obs(obs: Dict, batch_idx: torch.Tensor) -> Dict:
        """
        Sub-index a nuplan feature dict along the batch dimension.

        Handles arbitrarily nested dicts of tensors.  Non-tensor values
        are passed through unchanged (which is correct for scalars/strings).
        """

        def _index(v, idx):
            if isinstance(v, torch.Tensor):
                return v[idx]
            if isinstance(v, dict):
                return {kk: _index(vv, idx) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_index(x, idx) for x in v)
            return v

        return _index(obs, batch_idx)
