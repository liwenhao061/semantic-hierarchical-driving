from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class LLMSemanticEncoder(nn.Module):
    def __init__(
        self, 
        dim=128, 
        llm_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_dim=384,
        freeze_llm=True
    ):
        super().__init__()
        self.dim = dim
        self.llm_dim = llm_dim
        self.freeze_llm = freeze_llm
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModel.from_pretrained(llm_model_name)
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(llm_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def train(self, mode=True):
        """Keep the frozen language backbone deterministic during training."""
        super().train(mode)
        if self.freeze_llm:
            self.llm.eval()
        return self
        
    def forward(self, scenario_descriptions):
        if isinstance(scenario_descriptions, str):
            scenario_descriptions = [scenario_descriptions]
            
        tokens = self.tokenizer(
            scenario_descriptions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(next(self.llm.parameters()).device)
        
        grad_context = torch.no_grad() if self.freeze_llm else nullcontext()
        with grad_context:
            llm_outputs = self.llm(**tokens)
            embeddings = llm_outputs.last_hidden_state[:, 0]
        
        semantic_features = self.semantic_proj(embeddings)
        return semantic_features
    
    def generate_scenario_description(self, data):
        bs = data["agent"]["position"].shape[0]
        descriptions = []
        
        for b in range(bs):
            agent_mask = data["agent"]["valid_mask"][b, :, -1]
            num_agents = agent_mask.sum().item() - 1
            
            ego_vel = torch.norm(data["agent"]["velocity"][b, 0, -1]).item()
            
            map_on_route = data["map"]["polygon_on_route"][b]
            tl_status = data["map"]["polygon_tl_status"][b]
            
            has_red_light = ((tl_status == 1) & map_on_route).any().item()
            has_green_light = ((tl_status == 2) & map_on_route).any().item()
            
            if ego_vel < 1.0:
                speed_desc = "stopped"
            elif ego_vel < 5.0:
                speed_desc = "moving slowly"
            else:
                speed_desc = "moving at normal speed"
            
            desc = f"Ego vehicle is {speed_desc} with {num_agents} surrounding agents."
            
            if has_red_light:
                desc += " Red traffic light ahead."
            elif has_green_light:
                desc += " Green traffic light."
                
            if num_agents > 5:
                desc += " Dense traffic environment."
            elif num_agents > 0:
                desc += " Moderate traffic."
            else:
                desc += " Empty road."
            
            descriptions.append(desc)
        
        return descriptions
