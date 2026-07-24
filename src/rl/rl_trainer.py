"""
RLTrainer
=========
Orchestrates the full RL closed-loop training pipeline:

  ┌──────────────────────────────────────────────────────────────────┐
  │  for each RL epoch:                                              │
  │    1. Rollout collection                                         │
  │       • iterate data_loader (nuplan batches)                     │
  │       • env.reset(batch)  →  obs                                 │
  │       • actor_critic.act(obs)  →  action, log_prob, value, ...   │
  │       • env.step(action, out)  →  reward, cost, done             │
  │       • buffer.add(...)                                          │
  │    2. buffer.finish_rollout(bootstrap_value)                     │
  │    3. PPO update                                                 │
  │       • ppo.update(buffer, λ, il_loss_fn)                        │
  │    4. CMDP dual update                                           │
  │       • cmdp.update(mean_cost_components)                        │
  │    5. Optional: refresh teacher demonstrations                   │
  │    6. Logging / checkpointing                                    │
  └──────────────────────────────────────────────────────────────────┘

The trainer can run in three modes (set via ``mode`` param):
  "rl_only"       – pure PPO+CMDP, no IL component
  "il_only"       – supervised IL (identical to the existing LightningTrainer)
  "rl_with_il"    – PPO+CMDP + IL distillation loss (recommended)
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.models.pluto.modules.actor_critic import ActorCriticWrapper
from .nuplan_env_wrapper import NuplanEnvWrapper
from .rollout_buffer import RolloutBuffer
from .ppo_updater import PPOUpdater
from .cmdp_dual_update import CMDPDualUpdater, ConstraintSpec
from .teacher_rollout_collector import TeacherRolloutCollector, DemonstrationBuffer

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Full RL training loop.

    Parameters
    ----------
    model                : ActorCriticWrapper (wraps PlanningModel)
    teacher_model        : Optional PlanningModel for teacher demonstrations.
                           May be the same object as the base of ``model``
                           (self-distillation) or a separate heavier model.
    optimizer            : parameter optimizer
    scheduler            : LR scheduler (called per epoch)
    env                  : NuplanEnvWrapper
    rollout_buffer       : RolloutBuffer
    ppo_updater          : PPOUpdater
    cmdp_updater         : CMDPDualUpdater
    teacher_collector    : TeacherRolloutCollector (optional)
    mode                 : "rl_only" | "il_only" | "rl_with_il"
    rollout_steps        : number of env steps per rollout before an update
    minibatch_size       : PPO mini-batch size
    il_loss_weight       : weight of IL/distillation loss in total PPO loss
    teacher_collect_freq : refresh teacher demo buffer every N epochs
    max_demo_batches     : max batches to collect per teacher refresh
    checkpoint_dir       : directory to save checkpoints
    log_freq             : log every N rollout steps
    device               : torch device
    """

    def __init__(
        self,
        model: ActorCriticWrapper,
        optimizer: Optimizer,
        env: NuplanEnvWrapper,
        rollout_buffer: RolloutBuffer,
        ppo_updater: PPOUpdater,
        cmdp_updater: CMDPDualUpdater,
        teacher_model: Optional[nn.Module] = None,
        teacher_collector: Optional[TeacherRolloutCollector] = None,
        scheduler: Optional[_LRScheduler] = None,
        mode: str = "rl_with_il",
        rollout_steps: int = 64,
        minibatch_size: int = 32,
        il_loss_weight: float = 0.1,
        teacher_collect_freq: int = 5,
        max_demo_batches: int = 100,
        checkpoint_dir: Optional[str] = None,
        log_freq: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.env = env
        self.buffer = rollout_buffer
        self.ppo = ppo_updater
        self.cmdp = cmdp_updater
        self.teacher_collector = teacher_collector
        self.mode = mode
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.il_loss_weight = il_loss_weight
        self.teacher_collect_freq = teacher_collect_freq
        self.max_demo_batches = max_demo_batches
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_freq = log_freq
        self.device = device or torch.device("cpu")

        self._demo_buffer: Optional[DemonstrationBuffer] = None
        self._global_step = 0
        self._epoch = 0

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(
        self,
        data_loader,
        num_epochs: int = 20,
        val_loader=None,
    ) -> Dict[str, List[float]]:
        """
        Run ``num_epochs`` epochs of RL training.

        Args:
            data_loader  : nuplan DataLoader returning (features, targets, scenarios)
            num_epochs   : total training epochs
            val_loader   : optional validation loader (for IL metrics)

        Returns:
            history dict  {metric_name: [value_per_epoch]}
        """
        history: Dict[str, List[float]] = {}

        for epoch in range(num_epochs):
            self._epoch = epoch
            logger.info("=== RL Epoch %d / %d ===", epoch + 1, num_epochs)

            # ----------------------------------------------------------------
            # 0. Optionally refresh teacher demonstrations
            # ----------------------------------------------------------------
            if (
                self.mode in ("rl_with_il", "il_only")
                and self.teacher_collector is not None
                and epoch % self.teacher_collect_freq == 0
            ):
                logger.info("Refreshing teacher demonstration buffer …")
                self._demo_buffer = self.teacher_collector.collect(
                    iter(data_loader),
                    max_batches=self.max_demo_batches,
                )

            # ----------------------------------------------------------------
            # 1. Rollout collection
            # ----------------------------------------------------------------
            if self.mode != "il_only":
                rollout_metrics = self._collect_rollouts(data_loader)
                self._update_history(history, rollout_metrics, prefix="rollout")
                logger.info("Rollout: %s", self._fmt(rollout_metrics))

            # ----------------------------------------------------------------
            # 2. PPO + CMDP update
            # ----------------------------------------------------------------
            if self.mode != "il_only":
                update_metrics = self._do_ppo_update()
                self._update_history(history, update_metrics, prefix="ppo")
                logger.info("PPO update: %s", self._fmt(update_metrics))

            # ----------------------------------------------------------------
            # 3. IL-only pass (if mode == "il_only")
            # ----------------------------------------------------------------
            if self.mode == "il_only":
                il_metrics = self._do_il_pass(data_loader)
                self._update_history(history, il_metrics, prefix="il")

            # ----------------------------------------------------------------
            # 4. LR scheduler step
            # ----------------------------------------------------------------
            if self.scheduler is not None:
                self.scheduler.step()

            # ----------------------------------------------------------------
            # 5. Checkpoint
            # ----------------------------------------------------------------
            if self.checkpoint_dir and (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)

        return history

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self, data_loader) -> Dict[str, float]:
        """
        Collect ``rollout_steps`` environment interactions and fill the buffer.
        """
        self.buffer.reset()
        self.model.eval()

        total_reward = 0.0
        total_cost = 0.0
        steps_done = 0
        cost_accumulator: Dict[str, List[float]] = {}

        data_iter = iter(data_loader)

        with torch.no_grad():
            while steps_done < self.rollout_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)

                features, targets, scenarios = batch
                data = self._to_device(features["feature"].data)

                # ── env.reset ──────────────────────────────────────────────
                obs, info = self.env.reset(data)

                # ── actor forward pass ────────────────────────────────────
                out = self.model.act(obs)

                action = out["selected_idx"]          # (bs,)
                log_prob = out["log_prob"]             # (bs,)
                value = out["value"]                  # (bs,)
                cost_value = out["cost_value"]        # (bs,)

                # ── env.step ──────────────────────────────────────────────
                # Extract semantic constraints if available
                sc = None
                if "constraint_info" in out and out["constraint_info"] is not None:
                    sc = out["constraint_info"].get("semantic_constraints")

                _, reward, cost, done, step_info = self.env.step(action, out, sc)

                # Accumulate cost components for CMDP dual update
                for name, val in step_info.cost_components.items():
                    cost_accumulator.setdefault(name, []).append(val)

                # ── buffer.add ────────────────────────────────────────────
                self.buffer.add(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    cost=cost,
                    value=value,
                    cost_value=cost_value,
                    done=done,
                )

                total_reward += reward.mean().item()
                total_cost += cost.mean().item()
                steps_done += 1
                self._global_step += 1

                if steps_done % self.log_freq == 0:
                    logger.debug(
                        "Step %d/%d  reward=%.4f  cost=%.4f",
                        steps_done, self.rollout_steps,
                        reward.mean().item(), cost.mean().item(),
                    )

        # ── bootstrap value at the end of the rollout ─────────────────────
        with torch.no_grad():
            try:
                final_batch = next(data_iter)
                final_data = self._to_device(final_batch[0]["feature"].data)
                final_out = self.model.act(final_data)
                last_value = final_out["value"]
                last_cost_value = final_out["cost_value"]
            except StopIteration:
                bs = self.buffer._bs or 1
                last_value = torch.zeros(bs, device=self.device)
                last_cost_value = torch.zeros(bs, device=self.device)

        self.buffer.finish_rollout(last_value, last_cost_value)

        # ── CMDP dual update ──────────────────────────────────────────────
        mean_costs = {k: sum(v) / len(v) for k, v in cost_accumulator.items()}
        lambda_update = self.cmdp.update(mean_costs)
        logger.debug("CMDP λ update: %s", lambda_update)

        return {
            "mean_reward": total_reward / max(steps_done, 1),
            "mean_cost": total_cost / max(steps_done, 1),
            **{f"cost_{k}": v for k, v in mean_costs.items()},
        }

    # ------------------------------------------------------------------
    # PPO update pass
    # ------------------------------------------------------------------

    def _do_ppo_update(self) -> Dict[str, float]:
        """Run PPO update on the filled rollout buffer."""
        self.model.train()
        lambdas = self.cmdp.get_lambda_tensor()

        # Build IL loss function (closed over demo_buffer)
        il_loss_fn = None
        if (
            self.mode == "rl_with_il"
            and self._demo_buffer is not None
            and self._demo_buffer.is_ready()
        ):
            demo_buffer = self._demo_buffer
            teacher_collector = self.teacher_collector
            model = self.model

            def il_loss_fn(sub_obs: Dict) -> torch.Tensor:
                demos = demo_buffer.sample(4)
                if not demos:
                    return torch.tensor(0.0, device=self.device)
                return teacher_collector.compute_il_loss(model, demos)

        ppo_metrics = self.ppo.update(
            buffer=self.buffer,
            lagrange_multipliers=lambdas,
            minibatch_size=self.minibatch_size,
            il_loss_fn=il_loss_fn,
            il_loss_weight=self.il_loss_weight,
        )
        return ppo_metrics

    # ------------------------------------------------------------------
    # IL-only pass (no RL, just supervised distillation)
    # ------------------------------------------------------------------

    def _do_il_pass(self, data_loader) -> Dict[str, float]:
        """
        One epoch of pure IL (teacher-student distillation) without RL.
        Falls back to a standard forward+loss loop if no demo buffer.
        """
        self.model.train()
        if (
            self._demo_buffer is None
            or not self._demo_buffer.is_ready()
            or self.teacher_collector is None
        ):
            logger.warning("IL pass skipped: demo buffer is empty.")
            return {"il_loss": 0.0}

        total_il_loss = 0.0
        steps = 0
        for batch in data_loader:
            demos = self._demo_buffer.sample(min(4, len(self._demo_buffer)))
            if not demos:
                break
            loss = self.teacher_collector.compute_il_loss(self.model, demos)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_il_loss += loss.item()
            steps += 1

        return {"il_loss": total_il_loss / max(steps, 1)}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / f"rl_checkpoint_epoch{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self._global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cmdp_lambdas": self.cmdp.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "cmdp_lambdas" in ckpt:
            self.cmdp.load_state_dict(ckpt["cmdp_lambdas"])
        self._epoch = ckpt.get("epoch", 0)
        self._global_step = ckpt.get("global_step", 0)
        logger.info("Checkpoint loaded from %s (epoch %d)", path, self._epoch)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_device(self, data: Dict) -> Dict:
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = self._to_device(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _update_history(
        history: Dict[str, List[float]],
        metrics: Dict[str, float],
        prefix: str = "",
    ) -> None:
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            history.setdefault(key, []).append(v)

    @staticmethod
    def _fmt(d: Dict[str, float]) -> str:
        return "  ".join(f"{k}={v:.4f}" for k, v in d.items())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rl_trainer(
    model: nn.Module,
    teacher_model: Optional[nn.Module],
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    warmup_epochs: int = 2,
    mode: str = "rl_with_il",
    rollout_steps: int = 64,
    minibatch_size: int = 32,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 4,
    il_loss_weight: float = 0.1,
    buffer_capacity: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    constraint_specs: Optional[List[ConstraintSpec]] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> "RLTrainer":
    """
    Convenience factory to assemble all RL components.

    Returns a fully configured RLTrainer ready to call .train(data_loader).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap in ActorCriticWrapper if not already wrapped
    if not isinstance(model, ActorCriticWrapper):
        ac_model = ActorCriticWrapper(model).to(device)
    else:
        ac_model = model.to(device)

    optimizer = torch.optim.AdamW(
        ac_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    from src.optim.warmup_cos_lr import WarmupCosLR
    scheduler = WarmupCosLR(
        optimizer=optimizer,
        lr=lr,
        min_lr=1e-6,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    env = NuplanEnvWrapper(device=device)
    buffer = RolloutBuffer(
        capacity=buffer_capacity,
        gamma=gamma,
        lam=gae_lambda,
        device=device,
    )
    ppo = PPOUpdater(
        model=ac_model,
        optimizer=optimizer,
        clip_eps=ppo_clip,
        ppo_epochs=ppo_epochs,
    )
    cmdp = CMDPDualUpdater(
        constraint_specs=constraint_specs,
        device=device,
    )

    tc = None
    if teacher_model is not None and mode in ("rl_with_il", "il_only"):
        tc = TeacherRolloutCollector(
            teacher_model=teacher_model,
            device=device,
        )

    return RLTrainer(
        model=ac_model,
        optimizer=optimizer,
        env=env,
        rollout_buffer=buffer,
        ppo_updater=ppo,
        cmdp_updater=cmdp,
        teacher_model=teacher_model,
        teacher_collector=tc,
        scheduler=scheduler,
        mode=mode,
        rollout_steps=rollout_steps,
        minibatch_size=minibatch_size,
        il_loss_weight=il_loss_weight,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
