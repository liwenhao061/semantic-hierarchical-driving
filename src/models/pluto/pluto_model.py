from copy import deepcopy
import math

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer
from .modules.llm_semantic_encoder import LLMSemanticEncoder
from .modules.hybrid_moe import HybridMoE
from .modules.semantic_safety_constraint import CMDPConstraintModule

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
        use_llm_rl_fusion=False,
        num_experts=4,
        use_semantic_constraints=False,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj
        self.use_llm_rl_fusion = use_llm_rl_fusion
        self.use_semantic_constraints = use_semantic_constraints

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)
        
        # Initialize the trainable planning backbone before constructing the
        # pretrained language model, otherwise Module.apply would overwrite the
        # frozen MiniLM weights.
        self.apply(self._init_weights)

        if self.use_llm_rl_fusion:
            self.llm_encoder = LLMSemanticEncoder(dim=dim)
            self.llm_encoder.semantic_proj.apply(self._init_weights)
            self.hybrid_moe = HybridMoE(
                dim=dim,
                num_experts=num_experts,
                num_modes=num_modes,
                future_steps=future_steps
            )
            self.hybrid_moe.apply(self._init_weights)

        if self.use_semantic_constraints:
            self.constraint_module = CMDPConstraintModule(dim=dim)
            self.constraint_module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        prediction = self.agent_predictor(x[:, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0
        
        semantic_features = None
        moe_gates = None
        moe_trajectory = None
        moe_hidden = None
        constraint_info = None
        
        if self.use_llm_rl_fusion:
            scenario_descriptions = self.llm_encoder.generate_scenario_description(data)
            semantic_features = self.llm_encoder(scenario_descriptions)
            
            moe_trajectory, moe_hidden, moe_gates, gate_logits = self.hybrid_moe(
                x[:, 0], semantic_features
            )
            
            if self.use_semantic_constraints:
                agent_positions = data["agent"]["position"][:, 1:A, self.history_steps-1]
                agent_valid = data["agent"]["valid_mask"][:, 1:A, self.history_steps-1]
                
                constraint_info = self.constraint_module(
                    semantic_features,
                    moe_hidden,
                    moe_trajectory[:, 0],
                    agent_positions,
                    agent_valid
                )

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None
        
        if self.use_llm_rl_fusion and not ref_line_available and moe_trajectory is not None:
            trajectory = moe_trajectory.unsqueeze(1)
            bs = trajectory.shape[0]
            probability = torch.full(
                (bs, 1, self.num_modes),
                1.0 / self.num_modes,
                device=trajectory.device,
            )

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj
        
        if self.use_llm_rl_fusion:
            if self.training:
                out["semantic_features"] = semantic_features
                out["moe_gates"] = moe_gates
                out["moe_trajectory"] = moe_trajectory
                out["moe_hidden"] = moe_hidden
            else:
                # moe_hidden is needed by ActorCriticWrapper during rollout (eval mode)
                out["moe_hidden"] = moe_hidden
                out["moe_gates"] = moe_gates
            
            if not self.training and moe_trajectory is not None:
                out["moe_output_trajectory"] = moe_trajectory
            
        if self.use_semantic_constraints and constraint_info is not None:
            out["constraint_info"] = constraint_info

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if ref_line_available and trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                if self.use_llm_rl_fusion and moe_trajectory is not None:
                    moe_angle = torch.atan2(moe_trajectory[..., 3], moe_trajectory[..., 2])
                    moe_out_trajectory = torch.cat(
                        [moe_trajectory[..., :2], moe_angle.unsqueeze(-1)], dim=-1
                    )

                    best_trajectory = moe_out_trajectory[:, 0]
                    out["output_trajectory"] = best_trajectory
                    out["candidate_trajectories"] = moe_out_trajectory.unsqueeze(1)
                elif "output_ref_free_trajectory" in out:
                    out["output_trajectory"] = out["output_ref_free_trajectory"]
                    out["probability"] = torch.zeros(1, 0, 0)
                    out["candidate_trajectories"] = torch.zeros(
                        1, 0, 0, self.future_steps, 3
                    )
                else:
                    out["output_trajectory"] = torch.zeros(bs, self.future_steps, 3, device=x.device)
                    out["probability"] = torch.zeros(1, 0, 0)
                    out["candidate_trajectories"] = torch.zeros(
                        1, 0, 0, self.future_steps, 3
                    )

        return out
