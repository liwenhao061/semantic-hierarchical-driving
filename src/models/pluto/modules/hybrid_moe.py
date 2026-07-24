import torch
import torch.nn as nn
import torch.nn.functional as F


class RLExpert(nn.Module):
    def __init__(self, dim=128, num_modes=6, future_steps=80):
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        self.policy_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, num_modes * future_steps * 4)
        )
        
    def forward(self, x):
        h = self.policy_head(x)
        traj = self.trajectory_decoder(h)
        bs = x.shape[0]
        traj = traj.reshape(bs, self.num_modes, self.future_steps, 4)
        return traj, h


class SemanticGatingModule(nn.Module):
    def __init__(self, dim=128, num_experts=4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, state_features, semantic_features):
        combined = torch.cat([state_features, semantic_features], dim=-1)
        logits = self.gate_net(combined)
        
        if self.training:
            gates = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            gates = F.softmax(logits / self.temperature, dim=-1)
            
        return gates, logits


class HybridMoE(nn.Module):
    def __init__(
        self, 
        dim=128, 
        num_experts=4, 
        num_modes=6, 
        future_steps=80,
        top_k=2
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.top_k = top_k
        
        self.experts = nn.ModuleList([
            RLExpert(dim, num_modes, future_steps) 
            for _ in range(num_experts)
        ])
        
        self.gating = SemanticGatingModule(dim, num_experts)
        
        self.expert_specialization = nn.Parameter(
            torch.randn(num_experts, dim) * 0.02
        )
        
    def forward(self, state_features, semantic_features):
        gates, gate_logits = self.gating(state_features, semantic_features)
        
        bs = state_features.shape[0]
        expert_outputs = []
        expert_hiddens = []
        
        for expert in self.experts:
            traj, h = expert(state_features)
            expert_outputs.append(traj)
            expert_hiddens.append(h)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_hiddens = torch.stack(expert_hiddens, dim=1)
        
        if self.training:
            combined_traj = torch.einsum('be,bemtc->bmtc', gates, expert_outputs)
            combined_hidden = torch.einsum('be,bed->bd', gates, expert_hiddens)
        else:
            top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1)
            top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
            
            combined_traj = torch.zeros(
                bs, self.num_modes, self.future_steps, 4,
                device=state_features.device
            )
            combined_hidden = torch.zeros(bs, self.dim, device=state_features.device)
            
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                weight = top_k_gates[:, i]
                
                for b in range(bs):
                    combined_traj[b] += weight[b] * expert_outputs[b, expert_idx[b]]
                    combined_hidden[b] += weight[b] * expert_hiddens[b, expert_idx[b]]
        
        return combined_traj, combined_hidden, gates, gate_logits

