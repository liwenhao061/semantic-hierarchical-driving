import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSafetyPrior(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        
        self.constraint_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 4)
        )
        
        self.constraint_weight_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, semantic_features):
        constraints = self.constraint_generator(semantic_features)
        
        min_distance = F.softplus(constraints[:, 0]) + 2.0
        max_speed = F.softplus(constraints[:, 1]) + 5.0
        max_accel = F.softplus(constraints[:, 2]) + 2.0
        max_lat_accel = F.softplus(constraints[:, 3]) + 2.0
        
        constraint_weight = self.constraint_weight_net(semantic_features)
        
        return {
            'min_distance': min_distance,
            'max_speed': max_speed,
            'max_accel': max_accel,
            'max_lat_accel': max_lat_accel,
            'weight': constraint_weight
        }


class SemanticConsistencyScorer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        
        self.scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, semantic_features, action_features):
        combined = torch.cat([semantic_features, action_features], dim=-1)
        consistency_score = torch.sigmoid(self.scorer(combined))
        return consistency_score


class CMDPConstraintModule(nn.Module):
    def __init__(self, dim=128, num_constraint_types=5):
        super().__init__()
        self.dim = dim
        self.num_constraint_types = num_constraint_types
        
        self.safety_prior = SemanticSafetyPrior(dim)
        self.consistency_scorer = SemanticConsistencyScorer(dim)
        
        self.constraint_lambda = nn.Parameter(torch.ones(num_constraint_types))
        
    def compute_constraint_violations(
        self, 
        trajectory, 
        semantic_constraints,
        agent_positions,
        agent_mask
    ):
        if len(trajectory.shape) == 3:
            bs, T, C = trajectory.shape
        elif len(trajectory.shape) == 4:
            bs, M, T, C = trajectory.shape
            trajectory = trajectory[:, 0]
        else:
            raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")
        
        violations = {}
        
        ego_pos = trajectory[..., :2]
        ego_heading_vec = trajectory[..., 2:4] if trajectory.shape[-1] >= 4 else None
        
        speed = torch.norm(
            torch.diff(ego_pos, dim=1, prepend=ego_pos[:, :1]), 
            dim=-1
        ) / 0.1
        
        accel = torch.diff(speed, dim=1, prepend=speed[:, :1]) / 0.1
        
        if agent_positions is not None and agent_mask is not None and agent_positions.shape[1] > 0:
            agent_pos_expanded = agent_positions.unsqueeze(2)
            ego_pos_expanded = ego_pos.unsqueeze(1)
            
            distances = torch.norm(
                agent_pos_expanded - ego_pos_expanded, 
                dim=-1
            )
            distances = distances.masked_fill(~agent_mask.unsqueeze(-1), 1e6)
            min_distances = distances.min(dim=1)[0]
        else:
            min_distances = torch.ones(bs, T, device=trajectory.device) * 1e6
        
        violations['distance'] = F.relu(
            semantic_constraints['min_distance'].unsqueeze(-1) - min_distances
        )
        violations['speed'] = F.relu(
            speed - semantic_constraints['max_speed'].unsqueeze(-1)
        )
        violations['accel'] = F.relu(
            torch.abs(accel) - semantic_constraints['max_accel'].unsqueeze(-1)
        )
        
        return violations
    
    def forward(
        self, 
        semantic_features, 
        action_features, 
        trajectory,
        agent_positions=None,
        agent_mask=None
    ):
        semantic_constraints = self.safety_prior(semantic_features)
        
        consistency_score = self.consistency_scorer(semantic_features, action_features)
        
        violations = self.compute_constraint_violations(
            trajectory, 
            semantic_constraints,
            agent_positions,
            agent_mask
        )
        
        constraint_loss = 0.0
        for i, (name, violation) in enumerate(violations.items()):
            if i < self.num_constraint_types:
                weighted_violation = violation * semantic_constraints['weight']
                constraint_loss += self.constraint_lambda[i] * weighted_violation.mean()
        
        consistency_loss = (1.0 - consistency_score).mean()
        
        total_constraint_loss = constraint_loss + 0.1 * consistency_loss
        
        return {
            'constraint_loss': total_constraint_loss,
            'violations': violations,
            'consistency_score': consistency_score,
            'semantic_constraints': semantic_constraints
        }

