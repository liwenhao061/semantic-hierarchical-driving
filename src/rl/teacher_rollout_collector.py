"""
TeacherRolloutCollector
=======================
Collects trajectory rollouts from the *teacher* model (the full heavy
PlanningModel with LLM + MoE enabled) and stores them in a replay buffer
for use in student-policy distillation during RL training.

Design
------
The teacher is run in evaluation mode (no_grad) over a data loader.
For each batch the collector records:
  - teacher_trajectory  : (bs, T, 4)  – best teacher trajectory
  - teacher_probability : (bs, R×M)   – flattened softmax scores
  - teacher_hidden      : (bs, D)     – encoder / MoE hidden state
  - semantic_features   : (bs, D)     – LLM semantic embedding
  - data                : nuplan feature dict

These are stored in a simple list-based buffer (``DemonstrationBuffer``)
that the student RL policy samples from during the distillation phase.

Usage
-----
    collector = TeacherRolloutCollector(teacher_model, device=device)
    demo_buf  = collector.collect(data_loader, max_batches=200)

    # In RL trainer's IL loss function:
    il_loss = demo_buf.sample_il_loss(student_model, batch_size=32)
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demonstration buffer
# ---------------------------------------------------------------------------

@dataclass
class Demonstration:
    """One batch of teacher demonstrations."""
    data: Dict                              # nuplan feature dict (CPU tensors)
    trajectory: torch.Tensor               # (bs, T, 4)
    probability: torch.Tensor              # (bs, N_cand)
    hidden: torch.Tensor                   # (bs, D)
    semantic_features: Optional[torch.Tensor] = None  # (bs, D) or None


class DemonstrationBuffer:
    """
    Fixed-size circular buffer of teacher demonstrations.

    Parameters
    ----------
    max_size : maximum number of Demonstration entries to keep
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._data: List[Demonstration] = []
        self._ptr = 0

    def add(self, demo: Demonstration) -> None:
        if len(self._data) < self.max_size:
            self._data.append(demo)
        else:
            self._data[self._ptr] = demo
        self._ptr = (self._ptr + 1) % self.max_size

    def sample(self, n: int) -> List[Demonstration]:
        k = min(n, len(self._data))
        return random.sample(self._data, k)

    def __len__(self) -> int:
        return len(self._data)

    def is_ready(self) -> bool:
        return len(self._data) > 0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TeacherRolloutCollector:
    """
    Runs the teacher model over a dataset and fills a DemonstrationBuffer.

    Parameters
    ----------
    teacher_model : PlanningModel (or ActorCriticWrapper around it)
                    Must have use_llm_rl_fusion=True and use_hidden_proj=True
                    (or expose 'hidden'/'moe_hidden' in its output dict).
    device        : torch device
    history_steps : number of history steps (default 21, must match model)
    """

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        device: Optional[torch.device] = None,
        history_steps: int = 21,
    ) -> None:
        self.teacher = teacher_model
        self.device = device or torch.device("cpu")
        self.history_steps = history_steps

    @torch.no_grad()
    def collect(
        self,
        data_iterator: Iterator,
        max_batches: int = 200,
        buffer: Optional[DemonstrationBuffer] = None,
    ) -> DemonstrationBuffer:
        """
        Iterate over data_iterator and run the teacher on each batch.

        Args:
            data_iterator : iterable yielding (features, targets, scenarios)
                            as from a nuplan DataLoader
            max_batches   : stop after this many batches
            buffer        : existing DemonstrationBuffer to extend; if None
                            a new one is created

        Returns:
            DemonstrationBuffer filled with teacher demonstrations
        """
        if buffer is None:
            buffer = DemonstrationBuffer()

        self.teacher.eval()
        self.teacher.to(self.device)

        collected = 0
        for batch_idx, batch in enumerate(data_iterator):
            if collected >= max_batches:
                break

            features, targets, scenarios = batch
            data = self._move_to_device(features["feature"].data)

            out = self.teacher(data)

            # ----------------------------------------------------------------
            # Extract teacher trajectory
            # ----------------------------------------------------------------
            teacher_traj = self._extract_trajectory(out, data)
            if teacher_traj is None:
                continue

            # ----------------------------------------------------------------
            # Extract teacher probability
            # ----------------------------------------------------------------
            teacher_prob = self._extract_probability(out)

            # ----------------------------------------------------------------
            # Extract hidden state
            # ----------------------------------------------------------------
            hidden = self._extract_hidden(out)
            if hidden is None:
                continue

            # ----------------------------------------------------------------
            # Semantic features (optional)
            # ----------------------------------------------------------------
            semantic = out.get("semantic_features", None)

            demo = Demonstration(
                data=self._move_to_cpu(data),
                trajectory=teacher_traj.cpu(),
                probability=teacher_prob.cpu(),
                hidden=hidden.cpu(),
                semantic_features=semantic.cpu() if semantic is not None else None,
            )
            buffer.add(demo)
            collected += 1

            if collected % 50 == 0:
                logger.info("Collected %d teacher demonstrations", collected)

        logger.info(
            "Teacher rollout collection done. Buffer size: %d", len(buffer)
        )
        return buffer

    # ------------------------------------------------------------------
    # IL loss helper
    # ------------------------------------------------------------------

    def compute_il_loss(
        self,
        student_model: torch.nn.Module,
        demos: List[Demonstration],
        history_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the imitation (distillation) loss of ``student_model``
        against the stored teacher demonstrations.

        The student is expected to have a ``policy_distillation`` attribute
        (PolicyDistillation) as set up in LightningTrainer, or to expose
        'trajectory' and 'probability' in its forward output.

        Args:
            student_model  : model being trained (in training mode)
            demos          : list of Demonstration objects
            history_steps  : override self.history_steps

        Returns:
            scalar IL loss tensor
        """
        hist = history_steps or self.history_steps
        losses = []

        for demo in demos:
            data = self._move_to_device(demo.data)
            bs = demo.trajectory.shape[0]

            out_student = student_model(data)

            # Match shapes
            teacher_traj = demo.trajectory.to(self.device)
            teacher_prob = demo.probability.to(self.device)
            teacher_hidden = demo.hidden.to(self.device)

            student_traj = out_student.get("trajectory")
            student_prob = out_student.get("probability")
            student_hidden = out_student.get("hidden")
            if student_hidden is None:
                student_hidden = out_student.get("moe_hidden")

            if student_traj is None or student_hidden is None:
                continue

            # Trajectory distillation (smooth L1 on best candidate)
            if teacher_traj.dim() == 3:    # (bs, T, 4)
                t_traj = teacher_traj
            elif teacher_traj.dim() == 4:  # (bs, M, T, 4) → take mode 0
                t_traj = teacher_traj[:, 0]

            if student_traj.dim() == 5:    # (bs, R, M, T, 4) → take R=0, M=0
                s_traj = student_traj[:, 0, 0]
            elif student_traj.dim() == 4:  # (bs, M, T, 4)
                s_traj = student_traj[:, 0]
            else:
                s_traj = student_traj

            traj_loss = F.smooth_l1_loss(
                s_traj[..., :t_traj.shape[-1]],
                t_traj.detach(),
            )

            # Probability distillation (soft KL)
            if teacher_prob.dim() > 1 and student_prob is not None:
                t_prob_flat = teacher_prob.reshape(bs, -1)
                if student_prob.dim() == 3:
                    s_prob_flat = student_prob.reshape(bs, -1)
                else:
                    s_prob_flat = student_prob

                # Pad / trim to same width
                n = min(t_prob_flat.shape[-1], s_prob_flat.shape[-1])
                t_soft = F.softmax(t_prob_flat[:, :n] / 2.0, dim=-1)
                s_log_soft = F.log_softmax(s_prob_flat[:, :n] / 2.0, dim=-1)
                prob_loss = F.kl_div(s_log_soft, t_soft.detach(), reduction="batchmean") * 4.0
            else:
                prob_loss = torch.tensor(0.0, device=self.device)

            # Hidden-state alignment
            if student_hidden is not None:
                t_h = teacher_hidden
                s_h = student_hidden
                # Project to same size if needed
                if t_h.shape[-1] != s_h.shape[-1]:
                    n = min(t_h.shape[-1], s_h.shape[-1])
                    t_h = t_h[:, :n]
                    s_h = s_h[:, :n]
                hidden_loss = F.mse_loss(s_h, t_h.detach())
            else:
                hidden_loss = torch.tensor(0.0, device=self.device)

            losses.append(traj_loss + 0.5 * prob_loss + 0.3 * hidden_loss)

        if not losses:
            return torch.tensor(
                0.0, device=self.device, requires_grad=True
            )

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _move_to_device(self, data: Dict) -> Dict:
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = self._move_to_device(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _move_to_cpu(data: Dict) -> Dict:
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.cpu()
            elif isinstance(v, dict):
                result[k] = TeacherRolloutCollector._move_to_cpu(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _extract_trajectory(out: Dict, data: Dict) -> Optional[torch.Tensor]:
        """Return (bs, T, 4) trajectory from model output, or None."""
        if "moe_trajectory" in out and out["moe_trajectory"] is not None:
            t = out["moe_trajectory"]
            return t[:, 0] if t.dim() == 4 else t

        if "trajectory" in out and out["trajectory"] is not None:
            t = out["trajectory"]       # (bs, R, M, T, 4)
            if t.dim() == 5:
                prob = out.get("probability")
                if prob is not None:
                    bs = t.shape[0]
                    R, M = t.shape[1], t.shape[2]
                    flat_prob = prob.reshape(bs, R * M)
                    best = flat_prob.argmax(-1)
                    flat_t = t.reshape(bs, R * M, t.shape[3], t.shape[4])
                    return flat_t[torch.arange(bs, device=t.device), best]
                return t[:, 0, 0]       # fallback
            return t

        return None

    @staticmethod
    def _extract_probability(out: Dict) -> torch.Tensor:
        """Return a 2-D (bs, N_cand) probability tensor."""
        prob = out.get("probability")
        if prob is None:
            return torch.zeros(1, 1)
        bs = prob.shape[0]
        return prob.reshape(bs, -1)

    @staticmethod
    def _extract_hidden(out: Dict) -> Optional[torch.Tensor]:
        if "hidden" in out and out["hidden"] is not None:
            return out["hidden"]
        if "moe_hidden" in out and out["moe_hidden"] is not None:
            return out["moe_hidden"]
        return None
