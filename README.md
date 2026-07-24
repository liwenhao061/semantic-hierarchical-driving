# LLM-Augmented RL-CMDP Planner for Autonomous Driving

This project is developed based on [PLUTO](https://github.com/jchengai/pluto) (*Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving*) with semantic reasoning, constrained reinforcement learning, and policy distillation for autonomous driving planning.

Main features include:
- **Semantic encoder** (LLM sentence-transformer embedding)
- **MoE trajectory head** (Mixture-of-Experts with semantic gating)
- **CMDP constraint module** (Lagrangian safety constraints)
- **Policy distillation** (teacher-student deployment)
- **RL closed loop** (PPO + CMDP dual update + environment wrapper)

---

## Repository structure

```text
code/
├── run_training.py            # Supervised IL training entry point
├── run_rl_training.py         # RL closed-loop training entry point
├── config/
│   ├── default_training.yaml
│   ├── training/
│   │   ├── train_pluto.yaml          # baseline IL config
│   │   └── train_pluto_llm_rl.yaml   # IL + LLM + MoE + distillation
│   └── rl/
│       └── default_rl_training.yaml  # RL + CMDP config
└── src/
    ├── models/pluto/
    │   ├── pluto_model.py             # PlanningModel (full forward pass)
    │   ├── pluto_trainer.py           # LightningTrainer (IL training)
    │   ├── layers/                    # Fourier emb, Transformer, MLP
    │   ├── loss/
    │   │   └── esdf_collision_loss.py # ESDF-based collision loss
    │   └── modules/
    │       ├── agent_encoder.py
    │       ├── map_encoder.py
    │       ├── static_objects_encoder.py
    │       ├── planning_decoder.py
    │       ├── agent_predictor.py
    │       ├── llm_semantic_encoder.py         # [NEW] sentence-transformer encoder
    │       ├── hybrid_moe.py                   # [NEW] Mixture-of-Experts trajectory head
    │       ├── semantic_safety_constraint.py   # [NEW] CMDP constraint module
    │       ├── policy_distillation.py          # [NEW] teacher-to-student distillation
    │       └── actor_critic.py                 # [NEW] value head + log_prob for PPO
    └── rl/                                     # [NEW] RL closed-loop package
        ├── __init__.py
        ├── nuplan_env_wrapper.py               # gym-style env (step / reset)
        ├── rollout_buffer.py                   # PPO buffer + GAE (reward + cost)
        ├── ppo_updater.py                      # PPO actor-critic update
        ├── cmdp_dual_update.py                 # Lagrangian dual update + shielding
        ├── teacher_rollout_collector.py        # teacher demo collection
        └── rl_trainer.py                       # full RL training orchestrator
```

---

## Module descriptions

| Module | Role |
|--------|------|
| `LLMSemanticEncoder` | Encodes scenario description text via a frozen sentence-transformer and projects it into the model dimension |
| `HybridMoE` | Uses semantic and state features to gate multiple expert trajectory heads |
| `CMDPConstraintModule` | Generates scenario-level safety bounds and penalizes safety violations |
| `PolicyDistillation` | Trains a lightweight student policy from the teacher via trajectory and representation alignment |
| `ActorCriticWrapper` | Adds reward and cost value heads together with action log-probability computation |
| `NuplanEnvWrapper` | Converts nuPlan data batches into a gym-style `reset()` / `step()` interface |
| `RolloutBuffer` | Stores rollout tuples and computes GAE for both reward and safety cost |
| `PPOUpdater` | Performs PPO updates with CMDP Lagrangian penalty and optional IL regularization |
| `CMDPDualUpdater` | Updates Lagrange multipliers and supports trajectory shielding / projection |
| `TeacherRolloutCollector` | Collects teacher demonstrations for student distillation |
| `RLTrainer` | Orchestrates rollout, advantage estimation, PPO optimization, and dual-variable updates |

---

## Installation

```bash
pip install -r requirements.txt
# nuplan-devkit must already be installed:
# https://github.com/motional/nuplan-devkit
```

---

## 1. Supervised IL training (baseline)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto
```

## 2. IL training with LLM + MoE + distillation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto_llm_rl \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_llm_rl
```

## 3. RL closed-loop training (PPO + CMDP + teacher distillation)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_rl_training.py \
  py_func=rl_train +training=train_pluto_llm_rl \
  worker=single_machine_thread_pool worker.max_workers=16 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=16 data_loader.params.num_workers=8 \
  lr=3e-4 epochs=20 warmup_epochs=2 weight_decay=1e-4 \
  rl.mode=rl_with_il \
  rl.rollout_steps=64 \
  rl.ppo_epochs=4 \
  rl.il_loss_weight=0.1 \
  rl.checkpoint_dir=./outputs/rl_checkpoints \
  checkpoint=/path/to/il_pretrained.ckpt \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_rl
```

**Recommended workflow:** pretrain with step 2, then fine-tune with step 3.

### RL modes

| `rl.mode` | Description |
|-----------|-------------|
| `rl_only` | Pure PPO + CMDP, no IL loss |
| `il_only` | Supervised distillation from teacher only (no RL gradients) |
| `rl_with_il` | PPO + CMDP + teacher distillation regularisation **(recommended)** |

---

## 4. nuPlan simulation / evaluation

```bash
python run_simulation.py \
  +simulation=default_simulation \
  planner=pluto_planner \
  scenario_builder=nuplan \
  scenario_filter=test_scenarios \
  planner.pluto_planner.planner_ckpt=/path/to/checkpoint.pth
```

---

## CMDP constraint configuration

Constraint budgets and Lagrange multiplier learning rates are set in
`config/rl/default_rl_training.yaml` under `rl.constraints`.
The `CMDPDualUpdater` maintains one multiplier `λ_i` for each constraint and updates it at each rollout:

```text
λ_i ← max(0, λ_i + α_i · (mean_cost_i − threshold_i))
```

Trajectory **shielding** can be applied at inference time:

```python
from src.rl.cmdp_dual_update import CMDPDualUpdater
safe_traj = CMDPDualUpdater.shield_trajectory(trajectory, semantic_constraints)
```

---

## Key dependencies

```text
torch >= 2.0
pytorch-lightning >= 2.0
transformers >= 4.30
nuplan-devkit
hydra-core >= 1.3
omegaconf >= 2.3
```

------



## Release Status

The code is under cleaning and will be released gradually.

- [ ] improve docs
- [ ] training code
- [ ] simulation / evaluation code
- [ ] checkpoints and pretrained weights

------



## Citation

If you find this repository useful, please cite our work:

```bibtex
@unpublished{li2026semantics,
  title  = {Semantics-Guided Hierarchical Decision-Making for Autonomous Driving via {LLM}-Assisted Reinforcement Learning},
  author = {Li, Wenhao and Wang, Tao and Hu, Songhua},
  note   = {Manuscript under review at Transportation Research Part C: Emerging Technologies},
  year   = {2026}
}
```

The citation metadata will be updated with the DOI, volume, and page numbers after publication.
