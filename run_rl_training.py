"""
run_rl_training.py
==================
Entry point for RL closed-loop training.

Supports three training modes via --mode:
  rl_only      – PPO + CMDP, no IL component
  il_only      – supervised imitation / distillation (no RL gradients)
  rl_with_il   – PPO + CMDP + distillation from teacher (recommended)

Example invocation
------------------
# RL with teacher distillation, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_rl_training.py \\
  +rl=default_rl_training \\
  worker=single_machine_thread_pool worker.max_workers=16 \\
  scenario_builder=nuplan \\
  cache.cache_path=/nuplan/exp/cache_pluto_1M \\
  cache.use_cache_without_dataset=true \\
  data_loader.params.batch_size=16 data_loader.params.num_workers=8 \\
  lr=3e-4 epochs=20 warmup_epochs=2 weight_decay=1e-4 \\
  rl.mode=rl_with_il rl.rollout_steps=64 rl.ppo_epochs=4 \\
  rl.il_loss_weight=0.1 rl.checkpoint_dir=./outputs/rl_checkpoints \\
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_rl
"""

import logging
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig, OmegaConf

from src.custom_training import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)
from src.rl.rl_trainer import build_rl_trainer
from src.rl.cmdp_dual_update import build_constraint_specs
from src.models.pluto.modules.actor_critic import ActorCriticWrapper

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

set_default_path()

CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entry point for RL training / standard training / caching.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    build_logger(cfg)
    update_config_for_training(cfg)
    build_training_experiment_folder(cfg=cfg)
    worker = build_worker(cfg)

    # ----------------------------------------------------------------
    # cache
    # ----------------------------------------------------------------
    if cfg.py_func == "cache":
        logger.info("Starting caching …")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None

    # ----------------------------------------------------------------
    # RL training
    # ----------------------------------------------------------------
    if cfg.py_func == "rl_train":
        logger.info("Starting RL training …")

        # Build standard Lightning engine to get model + datamodule
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        requested_device = OmegaConf.select(cfg, "rl.device", default=None)
        if requested_device is not None:
            device = torch.device(str(requested_device))
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("rl.device requests CUDA, but CUDA is unavailable.")
        else:
            accelerator = str(
                OmegaConf.select(
                    cfg,
                    "lightning.trainer.params.accelerator",
                    default="auto",
                )
            ).lower()
            use_cuda = (
                accelerator in {"auto", "gpu", "cuda"}
                and torch.cuda.is_available()
            )
            device = torch.device("cuda" if use_cuda else "cpu")
        base_model = engine.model.model   # unwrap LightningTrainer → PlanningModel

        # Optional: teacher model = same base_model (self-distillation)
        teacher_model = base_model if getattr(cfg, "rl", None) and \
            OmegaConf.select(cfg, "rl.use_teacher", default=True) else None

        # RL hyper-parameters
        def select(path, default):
            return OmegaConf.select(cfg, path, default=default)

        def select_dict(path):
            value = OmegaConf.select(cfg, path, default=None)
            if value is None:
                return None
            return OmegaConf.to_container(value, resolve=True)

        mode = select("rl.mode", "rl_with_il")
        rollout_steps = int(select("rl.rollout_steps", 64))
        minibatch_size = int(select("rl.minibatch_size", 32))
        ppo_clip = float(select("rl.ppo_clip", 0.2))
        ppo_epochs = int(select("rl.ppo_epochs", 4))
        il_loss_weight = float(select("rl.il_loss_weight", 0.1))
        buffer_capacity = int(select("rl.buffer_capacity", 256))
        checkpoint_dir = select("rl.checkpoint_dir", None)

        gamma = float(select("rl.gamma", 0.99))
        gamma_cost = float(select("rl.gamma_cost", 0.99))
        gae_lambda = float(select("rl.gae_lambda", 0.95))
        value_coef = float(select("rl.value_coef", 0.5))
        cost_value_coef = float(select("rl.cost_value_coef", 0.5))
        entropy_coef = float(select("rl.entropy_coef", 0.01))
        max_grad_norm = float(select("rl.max_grad_norm", 0.5))
        target_kl = float(select("rl.target_kl", 0.01))
        teacher_collect_freq = int(select("rl.teacher_collect_freq", 5))
        max_demo_batches = int(select("rl.max_demo_batches", 100))

        reward_weights = select_dict("rl.reward_weights")
        cost_weights = select_dict("rl.cost_weights")
        constraint_specs = build_constraint_specs(
            select_dict("rl.constraints")
        )

        rl_trainer = build_rl_trainer(
            model=base_model,
            teacher_model=teacher_model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            mode=mode,
            rollout_steps=rollout_steps,
            minibatch_size=minibatch_size,
            ppo_clip=ppo_clip,
            ppo_epochs=ppo_epochs,
            il_loss_weight=il_loss_weight,
            buffer_capacity=buffer_capacity,
            gamma=gamma,
            gamma_cost=gamma_cost,
            gae_lambda=gae_lambda,
            lam_cost=gae_lambda,
            value_coef=value_coef,
            cost_value_coef=cost_value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            reward_weights=reward_weights,
            cost_weights=cost_weights,
            constraint_specs=constraint_specs,
            teacher_collect_freq=teacher_collect_freq,
            max_demo_batches=max_demo_batches,
            checkpoint_dir=checkpoint_dir or str(cfg.output_dir),
            device=device,
        )

        # Load IL checkpoint to initialise weights if specified
        if cfg.checkpoint:
            logger.info("Loading IL checkpoint: %s", cfg.checkpoint)
            state = torch.load(cfg.checkpoint, map_location=device)
            raw_state_dict = (
                state["state_dict"] if "state_dict" in state else state
            )
            model_state_dict = base_model.state_dict()
            state_dict = {}
            shape_mismatches = []
            for key, value in raw_state_dict.items():
                normalized_key = key.removeprefix("model.").removeprefix(
                    "base_model."
                )
                if normalized_key not in model_state_dict:
                    continue
                if model_state_dict[normalized_key].shape != value.shape:
                    shape_mismatches.append(normalized_key)
                    continue
                state_dict[normalized_key] = value

            if not state_dict:
                raise RuntimeError(
                    "Checkpoint contains no parameters matching PlanningModel."
                )
            if shape_mismatches:
                raise RuntimeError(
                    "Checkpoint tensor shapes do not match PlanningModel: "
                    + ", ".join(shape_mismatches[:10])
                )
            incompatible = base_model.load_state_dict(
                state_dict, strict=False
            )
            logger.info(
                "Loaded %d model tensors (%d missing, %d unexpected).",
                len(state_dict),
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

        # Get dataloaders from the Lightning datamodule
        datamodule = engine.datamodule
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None

        history = rl_trainer.train(
            data_loader=train_loader,
            num_epochs=cfg.epochs,
            val_loader=val_loader,
        )

        logger.info("RL training complete.")
        _log_final_history(history)
        return engine

    # ----------------------------------------------------------------
    # Standard supervised training (IL)
    # ----------------------------------------------------------------
    if cfg.py_func in ("train", "validate", "test"):
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        if cfg.py_func == "train":
            logger.info("Starting IL training …")
            with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
                engine.trainer.fit(
                    model=engine.model,
                    datamodule=engine.datamodule,
                    ckpt_path=cfg.checkpoint,
                )
        elif cfg.py_func == "validate":
            logger.info("Starting validation …")
            with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
                engine.trainer.validate(
                    model=engine.model,
                    datamodule=engine.datamodule,
                    ckpt_path=cfg.checkpoint,
                )
        elif cfg.py_func == "test":
            logger.info("Starting test …")
            with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
                engine.trainer.test(
                    model=engine.model,
                    datamodule=engine.datamodule,
                )
        return engine

    raise NameError(f"Unknown py_func: {cfg.py_func}")


def _log_final_history(history):
    logger.info("=== Training history (last epoch) ===")
    for k, v in history.items():
        if v:
            logger.info("  %-40s %.6f", k, v[-1])


if __name__ == "__main__":
    main()
