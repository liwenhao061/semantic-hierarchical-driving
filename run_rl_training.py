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
  py_func=rl_train +training=train_pluto_llm_rl \\
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = engine.model.model   # unwrap LightningTrainer → PlanningModel

        # Optional: teacher model = same base_model (self-distillation)
        teacher_model = base_model if getattr(cfg, "rl", None) and \
            OmegaConf.select(cfg, "rl.use_teacher", default=True) else None

        # RL hyper-parameters (with sane defaults)
        rl_cfg = OmegaConf.select(cfg, "rl") or {}
        mode = OmegaConf.select(cfg, "rl.mode", default="rl_with_il")
        rollout_steps = int(OmegaConf.select(cfg, "rl.rollout_steps", default=64))
        minibatch_size = int(OmegaConf.select(cfg, "rl.minibatch_size", default=32))
        ppo_clip = float(OmegaConf.select(cfg, "rl.ppo_clip", default=0.2))
        ppo_epochs = int(OmegaConf.select(cfg, "rl.ppo_epochs", default=4))
        il_loss_weight = float(OmegaConf.select(cfg, "rl.il_loss_weight", default=0.1))
        buffer_capacity = int(OmegaConf.select(cfg, "rl.buffer_capacity", default=256))
        checkpoint_dir = OmegaConf.select(cfg, "rl.checkpoint_dir", default=None)

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
            checkpoint_dir=checkpoint_dir or str(cfg.output_dir),
            device=device,
        )

        # Load IL checkpoint to initialise weights if specified
        if cfg.checkpoint:
            logger.info("Loading IL checkpoint: %s", cfg.checkpoint)
            state = torch.load(cfg.checkpoint, map_location=device)
            if "state_dict" in state:
                base_model.load_state_dict(state["state_dict"], strict=False)
            else:
                base_model.load_state_dict(state, strict=False)

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
