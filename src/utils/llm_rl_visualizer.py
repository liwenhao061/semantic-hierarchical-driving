import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


class LLMRLVisualizer:
    def __init__(self, save_dir: str = "./visualization"):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_expert_gates(self, gates: torch.Tensor, scenario_idx: int = 0):
        gates_np = gates[scenario_idx].detach().cpu().numpy()
        num_experts = len(gates_np)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(num_experts)
        bars = ax.bar(x, gates_np, alpha=0.7)
        
        for i, (bar, val) in enumerate(zip(bars, gates_np)):
            if val == gates_np.max():
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax.set_xlabel('Expert Index')
        ax.set_ylabel('Gate Weight')
        ax.set_title(f'Expert Selection Distribution (Scenario {scenario_idx})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Expert {i}' for i in range(num_experts)])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/expert_gates_{scenario_idx}.png', dpi=150)
        plt.close()
    
    def plot_constraint_violations(
        self, 
        violations: Dict[str, torch.Tensor], 
        scenario_idx: int = 0
    ):
        fig, axes = plt.subplots(len(violations), 1, figsize=(12, 3 * len(violations)))
        
        if len(violations) == 1:
            axes = [axes]
        
        for ax, (name, violation) in zip(axes, violations.items()):
            violation_np = violation[scenario_idx].detach().cpu().numpy()
            timesteps = np.arange(len(violation_np))
            
            ax.plot(timesteps, violation_np, linewidth=2)
            ax.fill_between(timesteps, 0, violation_np, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Violation Magnitude')
            ax.set_title(f'{name.capitalize()} Constraint Violations')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/constraint_violations_{scenario_idx}.png', dpi=150)
        plt.close()
    
    def plot_semantic_consistency(
        self, 
        consistency_scores: torch.Tensor,
        batch_size: Optional[int] = None
    ):
        scores_np = consistency_scores.detach().cpu().numpy().flatten()
        
        if batch_size is not None:
            scores_np = scores_np[:batch_size]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(scores_np))
        ax1.bar(x, scores_np, alpha=0.7)
        ax1.axhline(y=scores_np.mean(), color='r', linestyle='--', 
                   label=f'Mean: {scores_np.mean():.3f}')
        ax1.set_xlabel('Scenario Index')
        ax1.set_ylabel('Consistency Score')
        ax1.set_title('Semantic-Behavior Consistency Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(scores_np, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=scores_np.mean(), color='r', linestyle='--',
                   label=f'Mean: {scores_np.mean():.3f}')
        ax2.set_xlabel('Consistency Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Consistency Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/semantic_consistency.png', dpi=150)
        plt.close()
    
    def plot_distillation_comparison(
        self,
        teacher_traj: torch.Tensor,
        student_traj: torch.Tensor,
        scenario_idx: int = 0
    ):
        teacher_np = teacher_traj[scenario_idx, 0].detach().cpu().numpy()
        student_np = student_traj[scenario_idx, 0].detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        ax.plot(teacher_np[:, 0], teacher_np[:, 1], 'b-', linewidth=2, label='Teacher')
        ax.plot(student_np[:, 0], student_np[:, 1], 'r--', linewidth=2, label='Student')
        ax.scatter([0], [0], c='green', s=100, marker='o', label='Start')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        ax = axes[0, 1]
        error = np.linalg.norm(teacher_np[:, :2] - student_np[:, :2], axis=1)
        timesteps = np.arange(len(error))
        ax.plot(timesteps, error, linewidth=2)
        ax.fill_between(timesteps, 0, error, alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position Error (m)')
        ax.set_title(f'Position Error (Mean: {error.mean():.3f}m)')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        teacher_heading = np.arctan2(teacher_np[:, 3], teacher_np[:, 2])
        student_heading = np.arctan2(student_np[:, 3], student_np[:, 2])
        ax.plot(timesteps, teacher_heading, 'b-', linewidth=2, label='Teacher')
        ax.plot(timesteps, student_heading, 'r--', linewidth=2, label='Student')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Heading (rad)')
        ax.set_title('Heading Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        heading_error = np.abs(teacher_heading - student_heading)
        heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)
        ax.plot(timesteps, heading_error, linewidth=2)
        ax.fill_between(timesteps, 0, heading_error, alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Heading Error (rad)')
        ax.set_title(f'Heading Error (Mean: {heading_error.mean():.3f}rad)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/distillation_comparison_{scenario_idx}.png', dpi=150)
        plt.close()

