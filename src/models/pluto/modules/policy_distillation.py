import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightStudent(nn.Module):
    def __init__(
        self, 
        dim=64,
        state_channel=7,
        history_channel=7,
        history_steps=21,
        future_steps=80,
        num_modes=6
    ):
        super().__init__()
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.state_channel = state_channel
        self.history_channel = history_channel
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_channel, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
        self.history_encoder = nn.Sequential(
            nn.Linear(history_channel * history_steps, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
        self.trajectory_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, num_modes * future_steps * 4)
        )
        
        self.prob_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_modes)
        )
        
        self.semantic_interface = nn.Linear(dim, dim)
        
    def forward(self, state_features, history_features, semantic_correction=None):
        state_emb = self.state_encoder(state_features)
        
        bs = history_features.shape[0]
        history_flat = history_features.reshape(bs, -1)
        history_emb = self.history_encoder(history_flat)
        
        fused = self.fusion(torch.cat([state_emb, history_emb], dim=-1))
        
        if semantic_correction is not None:
            correction_emb = self.semantic_interface(semantic_correction)
            fused = fused + 0.1 * correction_emb
        
        trajectory = self.trajectory_head(fused)
        trajectory = trajectory.reshape(bs, self.num_modes, self.future_steps, 4)
        
        prob = self.prob_head(fused)
        
        return trajectory, prob, fused


class LanguageBehaviorAlignmentLoss(nn.Module):
    def __init__(self, dim=128, behavior_dim=None, temperature=0.5):
        super().__init__()
        behavior_dim = behavior_dim if behavior_dim is not None else dim
        self.dim = dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.language_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        self.behavior_projector = nn.Sequential(
            nn.Linear(behavior_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, semantic_features, behavior_features, labels=None):
        lang_proj = F.normalize(self.language_projector(semantic_features), dim=-1)
        behav_proj = F.normalize(self.behavior_projector(behavior_features), dim=-1)
        
        similarity = torch.matmul(lang_proj, behav_proj.t()) / self.temperature
        
        if labels is None:
            labels = torch.arange(semantic_features.size(0), device=semantic_features.device)
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class PolicyDistillation(nn.Module):
    def __init__(
        self,
        teacher_dim=128,
        student_dim=64,
        future_steps=80,
        num_modes=6,
        alpha_trajectory=1.0,
        alpha_probability=1.0,
        alpha_alignment=0.5,
        alpha_hidden=0.3
    ):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        
        self.alpha_trajectory = alpha_trajectory
        self.alpha_probability = alpha_probability
        self.alpha_alignment = alpha_alignment
        self.alpha_hidden = alpha_hidden
        
        self.student = LightweightStudent(
            dim=student_dim,
            future_steps=future_steps,
            num_modes=num_modes
        )
        
        self.alignment_loss = LanguageBehaviorAlignmentLoss(
            teacher_dim, behavior_dim=student_dim
        )
        
        self.hidden_adapter = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )
        
    def compute_trajectory_distillation_loss(self, student_traj, teacher_traj, valid_mask):
        if student_traj.shape != teacher_traj.shape:
            raise ValueError(
                "Trajectory shape mismatch: "
                f"student={tuple(student_traj.shape)}, "
                f"teacher={tuple(teacher_traj.shape)}"
            )

        loss = F.smooth_l1_loss(
            student_traj, teacher_traj.detach(), reduction='none'
        )
        loss = loss.sum(-1)
        
        if valid_mask is not None:
            mask = valid_mask.to(dtype=loss.dtype)
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_probability_distillation_loss(self, student_prob, teacher_prob, temperature=2.0):
        if student_prob.dim() != 2:
            raise ValueError(
                f"Expected student logits with shape (B, M), got {tuple(student_prob.shape)}"
            )

        if teacher_prob.dim() == 3:
            bs, num_references, num_modes = teacher_prob.shape
            teacher_soft = F.softmax(
                teacher_prob.detach().reshape(bs, num_references * num_modes)
                / temperature,
                dim=-1,
            ).reshape(bs, num_references, num_modes).sum(dim=1)
        elif teacher_prob.dim() == 2:
            teacher_soft = F.softmax(
                teacher_prob.detach() / temperature, dim=-1
            )
        else:
            raise ValueError(
                "Expected teacher logits with shape (B, M) or (B, R, M), "
                f"got {tuple(teacher_prob.shape)}"
            )

        if student_prob.shape != teacher_soft.shape:
            raise ValueError(
                "Probability mode mismatch after reference-line marginalization: "
                f"student={tuple(student_prob.shape)}, "
                f"teacher={tuple(teacher_soft.shape)}"
            )

        student_soft = F.log_softmax(student_prob / temperature, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        return kl_loss * (temperature ** 2)
    
    def compute_hidden_distillation_loss(self, student_hidden, teacher_hidden):
        student_adapted = self.hidden_adapter(student_hidden)
        loss = F.mse_loss(student_adapted, teacher_hidden.detach())
        return loss
    
    def forward(
        self, 
        student_inputs,
        teacher_outputs,
        semantic_features,
        valid_mask=None
    ):
        state_features = student_inputs['state']
        history_features = student_inputs['history']
        
        student_traj, student_prob, student_hidden = self.student(
            state_features, history_features
        )
        
        teacher_traj = teacher_outputs['trajectory']
        teacher_prob = teacher_outputs['probability']
        teacher_hidden = teacher_outputs['hidden']
        
        traj_loss = self.compute_trajectory_distillation_loss(
            student_traj, teacher_traj, valid_mask
        )
        
        prob_loss = self.compute_probability_distillation_loss(
            student_prob, teacher_prob
        )
        
        hidden_loss = self.compute_hidden_distillation_loss(
            student_hidden, teacher_hidden
        )
        
        alignment_loss = self.alignment_loss(semantic_features, student_hidden)
        
        total_loss = (
            self.alpha_trajectory * traj_loss +
            self.alpha_probability * prob_loss +
            self.alpha_alignment * alignment_loss +
            self.alpha_hidden * hidden_loss
        )
        
        return {
            'loss': total_loss,
            'traj_loss': traj_loss,
            'prob_loss': prob_loss,
            'alignment_loss': alignment_loss,
            'hidden_loss': hidden_loss,
            'student_traj': student_traj,
            'student_prob': student_prob
        }
