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
├── run_simulation.py          # nuPlan simulation / evaluation entry point
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

The tested setup uses Python 3.9, CUDA 11.8, PyTorch 2.0.1,
PyTorch Lightning 2.0.1, and NATTEN 0.14.6. Keep the non-PyTorch
dependencies installed by nuPlan-devkit, especially its Hydra/OmegaConf
versions; upgrading those packages independently can break nuPlan config
composition.

```bash
# Preserve the original nuPlan environment and upgrade only an isolated clone.
conda create --name nuplan-codev2 --clone nuplan -y
conda activate nuplan-codev2

# RTX 30 series uses compute capability 8.6; adjust for another GPU.
export TORCH_CUDA_ARCH_LIST=8.6
bash script/setup_env.sh

python - <<'PY'
import natten
import pytorch_lightning as pl
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, torch.cuda.is_available())
print("pytorch-lightning:", pl.__version__)
print("natten:", natten.__version__)
PY
```

`script/setup_env.sh` installs the CUDA 11.8 compiler/header packages from
NVIDIA's Conda channel and builds NATTEN from PyPI source. This avoids the
expired TLS certificate on the legacy NATTEN 0.14 wheel server.

Configure the nuPlan paths before caching, training, or simulation:

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset
export NUPLAN_MAPS_ROOT="${NUPLAN_DATA_ROOT}/maps"
export NUPLAN_DB_FILES="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/path/to/nuplan/exp
export PLUTO_CACHE="${NUPLAN_EXP_ROOT}/cache_pluto"
```

The `splits/mini` value is intended for smoke tests. Point
`NUPLAN_DB_FILES` at the required full split for production training or
benchmark evaluation.

If the cache does not exist yet, create it once:

```bash
python run_training.py \
  +training=train_pluto py_func=cache \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  cache.cache_path="${PLUTO_CACHE}" \
  cache.use_cache_without_dataset=false \
  wandb.mode=disable
```

The semantic encoder downloads
`sentence-transformers/all-MiniLM-L6-v2`; it does not call an external LLM
API. In a restricted network, download it once through an available mirror
and then train offline:

```bash
HF_ENDPOINT=https://hf-mirror.com python -c \
  "from transformers import AutoModel, AutoTokenizer; n='sentence-transformers/all-MiniLM-L6-v2'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"
export TRANSFORMERS_OFFLINE=1
```

---

## 1. Supervised IL training (baseline)

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  +training=train_pluto py_func=train \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  cache.cache_path="${PLUTO_CACHE}" \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=8 data_loader.params.num_workers=4 \
  lightning.trainer.params.devices=1 \
  lightning.trainer.params.strategy=auto \
  lightning.trainer.params.sync_batchnorm=false \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto
```

For a one-batch smoke test that still writes a checkpoint, replace the
training-length and data-loader overrides with:

```bash
data_loader.params.batch_size=1 data_loader.params.num_workers=0 \
lightning.trainer.params.limit_train_batches=1 \
lightning.trainer.params.limit_val_batches=1 \
lightning.trainer.params.num_sanity_val_steps=0 \
epochs=1 warmup_epochs=0 wandb.mode=disable
```

## 2. IL training with LLM + MoE + distillation

```bash
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python run_training.py \
  +training=train_pluto_llm_rl py_func=train \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  cache.cache_path="${PLUTO_CACHE}" \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=2 data_loader.params.num_workers=4 \
  lightning.trainer.params.devices=1 \
  lightning.trainer.params.strategy=auto \
  lightning.trainer.params.sync_batchnorm=false \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_llm_rl
```

Use the same checkpoint-producing one-batch overrides as step 1 when
validating a new environment.

## 3. RL closed-loop training (PPO + CMDP + teacher distillation)

```bash
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python run_rl_training.py \
  +rl=default_rl_training \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  cache.cache_path="${PLUTO_CACHE}" \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=1 data_loader.params.num_workers=4 \
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
The IL checkpoint must be a Lightning `.ckpt`; the loader removes its
`model.` key prefix before loading the underlying `PlanningModel`.

For a minimal end-to-end RL smoke test, use:

```bash
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python run_rl_training.py \
  +rl=default_rl_training \
  worker=single_machine_thread_pool worker.max_workers=2 \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  cache.cache_path="${PLUTO_CACHE}" \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=1 data_loader.params.num_workers=0 \
  epochs=1 warmup_epochs=0 \
  rl.mode=rl_with_il rl.rollout_steps=2 \
  rl.ppo_epochs=1 rl.minibatch_size=1 \
  rl.teacher_collect_freq=1 rl.max_demo_batches=1 \
  checkpoint=/path/to/llm_il.ckpt \
  wandb.mode=disable
```

### RL modes

| `rl.mode` | Description |
|-----------|-------------|
| `rl_only` | Pure PPO + CMDP, no IL loss |
| `il_only` | Supervised distillation from teacher only (no RL gradients) |
| `rl_with_il` | PPO + CMDP + teacher distillation regularisation **(recommended)** |

---

## 4. nuPlan simulation / evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run_simulation.py \
  +simulation=open_loop_boxes \
  planner=pluto_planner \
  scenario_builder=nuplan \
  scenario_builder.db_files="${NUPLAN_DB_FILES}" \
  scenario_filter=mini_demo_scenario \
  worker=sequential \
  number_of_gpus_allocated_per_simulation=1 \
  planner.pluto_planner.planner_ckpt=/path/to/baseline.ckpt
```

`mini_demo_scenario` is the one-scenario smoke test. Use the appropriate
benchmark scenario filter and full nuPlan split for a complete evaluation.
The default `pluto_planner` architecture matches the baseline model in step
1, so use a Lightning checkpoint produced by that step.

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
Python 3.9
torch 2.0.1+cu118
torchvision 0.15.2+cu118
pytorch-lightning 2.0.1
torchmetrics 0.10.2
natten 0.14.6 (CUDA build)
transformers 4.30.2
nuplan-devkit with its pinned hydra-core 1.1.0rc1 / omegaconf 2.1.0rc1
```

------



## Release Status

### Current release: Pre-API research code

This release contains the fully local and reproducible research pipeline:

- supervised IL baseline training;
- local MiniLM semantic encoding, MoE, and policy distillation;
- PPO + CMDP training with teacher distillation; and
- nuPlan simulation and open-loop evaluation.

No external OpenAI, Anthropic, or other hosted LLM inference API is used.
The semantic encoder runs
`sentence-transformers/all-MiniLM-L6-v2` locally after its weights have
been downloaded once. API credentials are therefore not required for the
current release.

- [x] improve docs
- [x] Pre-API training and RL code
- [x] nuPlan simulation / evaluation code
- [ ] checkpoints and pretrained weights

### Planned API-enabled version

An optional API-assisted version is planned as a separate future release.
It will add hosted-LLM semantic reasoning and constraint generation behind
explicit configuration, together with provider setup, credential handling,
response caching, offline fallbacks, and reproducibility notes. Those
components are intentionally **not included** in this Pre-API release, and
no release date is promised here.

------



## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{LI2026105891,
  title = {Semantics-guided hierarchical decision-making for autonomous driving via {LLM}-assisted reinforcement learning},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume = {192},
  pages = {105891},
  year = {2026},
  issn = {0968-090X},
  doi = {https://doi.org/10.1016/j.trc.2026.105891},
  url = {https://www.sciencedirect.com/science/article/pii/S0968090X26003773},
  author = {Wenhao Li and Tao Wang and Songhua Hu}
}
```
