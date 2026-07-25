import torch
import torch.nn as nn

from src.models.pluto.modules.actor_critic import ActorCriticWrapper
from src.models.pluto.modules.policy_distillation import PolicyDistillation
from src.planners.ml_planner_utils import load_checkpoint
from src.rl.cmdp_dual_update import build_constraint_specs
from src.rl.rollout_buffer import RolloutBuffer


def test_policy_distillation_accepts_reference_line_teacher_logits():
    module = PolicyDistillation(
        teacher_dim=8,
        student_dim=4,
        future_steps=3,
        num_modes=3,
    )
    batch_size = 2
    result = module(
        student_inputs={
            "state": torch.randn(batch_size, 7),
            "history": torch.randn(batch_size, 21, 7),
        },
        teacher_outputs={
            "trajectory": torch.randn(batch_size, 3, 3, 4),
            "probability": torch.randn(batch_size, 2, 3),
            "hidden": torch.randn(batch_size, 8),
        },
        semantic_features=torch.randn(batch_size, 8),
        valid_mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )

    assert torch.isfinite(result["loss"])
    result["loss"].backward()


def test_actor_critic_forward_delegates_to_base_model():
    class BaseModel(nn.Module):
        dim = 8

        def forward(self, data):
            return {"hidden": data["hidden"]}

    wrapper = ActorCriticWrapper(BaseModel())
    hidden = torch.randn(2, 8)

    assert wrapper({"hidden": hidden})["hidden"] is hidden


def test_singleton_rollout_advantages_are_finite():
    buffer = RolloutBuffer(capacity=1)
    scalar = torch.tensor([0.0])
    buffer.add(
        obs={"x": scalar},
        action=torch.tensor([0]),
        log_prob=scalar,
        reward=torch.tensor([1.0]),
        cost=scalar,
        value=scalar,
        cost_value=scalar,
        done=torch.tensor([True]),
    )
    buffer.finish_rollout(scalar, scalar)

    minibatch = next(buffer.iterate_minibatches(1, shuffle=False))
    assert torch.isfinite(minibatch["advantages"]).all()
    assert torch.isfinite(minibatch["cost_advantages"]).all()


def test_constraint_thresholds_are_wired_by_name():
    specs = build_constraint_specs(
        {
            "collision_threshold": 0.01,
            "speed_limit_threshold": 0.02,
        }
    )

    assert [(spec.name, spec.threshold) for spec in specs] == [
        ("collision", 0.01),
        ("speed_limit", 0.02),
    ]


def test_lightning_checkpoint_filters_non_model_state(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.tensor([1.0]),
                "policy_distillation.weight": torch.tensor([2.0]),
            }
        },
        checkpoint,
    )

    state_dict = load_checkpoint(str(checkpoint))
    assert list(state_dict) == ["weight"]
