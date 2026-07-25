"""
Actor-Critic extension for the PLUTO planning model.

Adds:
  - CriticValueHead   : scalar V(s) and cost-value Vc(s) estimates from encoder hidden state
  - PolicyLogProb     : log π(a|s) for a discrete trajectory-selection action
  - ActorCriticWrapper: wraps PlanningModel to expose actor-critic interface used by PPO
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticValueHead(nn.Module):
    """
    Two-head critic.
    - reward_value : V(s)  – expected discounted return
    - cost_value   : Vc(s) – expected discounted constraint cost (for CMDP)
    """

    def __init__(self, dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.reward_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.cost_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (bs, dim) – ego encoder token from the transformer
        Returns:
            value      : (bs,) reward value estimate
            cost_value : (bs,) cost value estimate
        """
        value = self.reward_head(hidden).squeeze(-1)
        cost_value = self.cost_head(hidden).squeeze(-1)
        return value, cost_value


class PolicyLogProb(nn.Module):
    """
    Converts the planning decoder's (trajectory, probability) output into a
    proper log-probability for the selected action (trajectory index).

    The policy is treated as a *categorical* distribution over candidate
    trajectories – the same discrete formulation used during inference.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        probability: torch.Tensor,
        selected_idx: torch.Tensor,
        r_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            probability   : (bs, R, M) or (bs, R*M) – raw logits from decoder
            selected_idx  : (bs,) – flat index of the chosen candidate
            r_padding_mask: (bs, R) bool mask for invalid reference lines

        Returns:
            log_prob: (bs,) log π(a|s) for the selected action
        """
        bs = probability.shape[0]
        if probability.dim() == 3:
            R, M = probability.shape[1], probability.shape[2]
            if r_padding_mask is not None:
                probability = probability.masked_fill(
                    r_padding_mask.unsqueeze(-1), -1e6
                )
            flat_prob = probability.reshape(bs, R * M)
        else:
            flat_prob = probability

        log_prob_all = F.log_softmax(flat_prob, dim=-1)
        log_prob = log_prob_all[torch.arange(bs, device=probability.device), selected_idx]
        return log_prob

    def entropy(
        self,
        probability: torch.Tensor,
        r_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Policy entropy H[π(·|s)] for entropy bonus in PPO."""
        bs = probability.shape[0]
        if probability.dim() == 3:
            R, M = probability.shape[1], probability.shape[2]
            if r_padding_mask is not None:
                probability = probability.masked_fill(
                    r_padding_mask.unsqueeze(-1), -1e6
                )
            flat_prob = probability.reshape(bs, R * M)
        else:
            flat_prob = probability

        dist = F.softmax(flat_prob, dim=-1)
        log_dist = F.log_softmax(flat_prob, dim=-1)
        ent = -(dist * log_dist).sum(dim=-1)
        return ent


class ActorCriticWrapper(nn.Module):
    """
    Thin wrapper that decorates PlanningModel with actor-critic functionality.

    The underlying ``base_model`` is the PlanningModel (or any model whose
    forward() returns a dict with 'trajectory', 'probability', optionally
    'hidden').  This wrapper adds:

      - critic      : CriticValueHead
      - log_prob_fn : PolicyLogProb

    Usage during rollout collection
    --------------------------------
    out = wrapper.act(data)          # forward pass + value + greedy action
    lp  = wrapper.log_prob(data, idx) # log π(a|s) for stored action

    Usage during PPO update
    -----------------------
    out   = wrapper.evaluate_actions(data, actions)
    # returns new log_probs, entropy, values, cost_values
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        dim = getattr(base_model, "dim", 128)

        self.critic = CriticValueHead(dim=dim)
        self.log_prob_fn = PolicyLogProb()

        self._init_critic()

    def _init_critic(self) -> None:
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_hidden(self, out: Dict, data: Dict) -> torch.Tensor:
        """
        Extract the ego encoder hidden state from the model output.
        Falls back to recomputing from the encoder if not cached.
        """
        if "hidden" in out:
            return out["hidden"]
        # moe_hidden is produced by HybridMoE and serves the same purpose
        if "moe_hidden" in out and out["moe_hidden"] is not None:
            return out["moe_hidden"]
        # Last resort: use the first token of the encoder (ego token)
        # This requires the base model to expose it – in PlanningModel
        # that token is x[:, 0] after the encoder blocks.
        raise RuntimeError(
            "ActorCriticWrapper: base model must output 'hidden' or 'moe_hidden'. "
            "Set use_hidden_proj=True or use_llm_rl_fusion=True."
        )

    def _get_r_padding_mask(self, data: Dict) -> Optional[torch.Tensor]:
        if "reference_line" in data and "valid_mask" in data["reference_line"]:
            return ~data["reference_line"]["valid_mask"].any(-1)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, data: Dict) -> Dict:
        """Delegate a standard forward pass to the wrapped planning model."""
        return self.base_model(data)

    def act(
        self, data: Dict
    ) -> Dict:
        """
        Forward pass + value estimation + greedy action selection.

        Returns dict with all base model outputs plus:
          'value'       : (bs,) V(s)
          'cost_value'  : (bs,) Vc(s)
          'selected_idx': (bs,) flat candidate index chosen by argmax
          'log_prob'    : (bs,) log π(a|s) for the chosen action
        """
        out = self.base_model(data)
        hidden = self._get_hidden(out, data)
        value, cost_value = self.critic(hidden)

        probability = out.get("probability")
        r_padding_mask = self._get_r_padding_mask(data)

        if probability is not None:
            bs = probability.shape[0]
            if probability.dim() == 3:
                R, M = probability.shape[1], probability.shape[2]
                if r_padding_mask is not None:
                    flat_prob = probability.masked_fill(
                        r_padding_mask.unsqueeze(-1), -1e6
                    ).reshape(bs, R * M)
                else:
                    flat_prob = probability.reshape(bs, R * M)
            else:
                flat_prob = probability
            selected_idx = flat_prob.argmax(dim=-1)
            log_prob = self.log_prob_fn(probability, selected_idx, r_padding_mask)
        else:
            # MoE-only mode: treat as single-action (idx 0)
            bs = hidden.shape[0]
            selected_idx = torch.zeros(bs, dtype=torch.long, device=hidden.device)
            log_prob = torch.zeros(bs, device=hidden.device)

        out["value"] = value
        out["cost_value"] = cost_value
        out["selected_idx"] = selected_idx
        out["log_prob"] = log_prob
        return out

    def evaluate_actions(
        self, data: Dict, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the current policy.
        Used during the PPO gradient update step.

        Args:
            data   : feature dict (same format as act())
            actions: (bs,) flat candidate indices from the rollout buffer

        Returns:
            log_prob   : (bs,)
            entropy    : (bs,)
            value      : (bs,)
            cost_value : (bs,)
        """
        out = self.base_model(data)
        hidden = self._get_hidden(out, data)
        value, cost_value = self.critic(hidden)

        probability = out.get("probability")
        r_padding_mask = self._get_r_padding_mask(data)

        if probability is not None:
            log_prob = self.log_prob_fn(probability, actions, r_padding_mask)
            entropy = self.log_prob_fn.entropy(probability, r_padding_mask)
        else:
            bs = hidden.shape[0]
            log_prob = torch.zeros(bs, device=hidden.device)
            entropy = torch.zeros(bs, device=hidden.device)

        return log_prob, entropy, value, cost_value
