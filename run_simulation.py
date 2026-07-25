"""Repository entry point for nuPlan closed-loop simulation and evaluation."""

import logging

import hydra
from nuplan.planning.script.run_simulation import run_simulation
from nuplan.planning.script.utils import set_default_path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)

set_default_path()

CONFIG_PATH = "./config"
CONFIG_NAME = "default_simulation"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """Run nuPlan simulation with this repository's planner configuration."""
    if cfg.simulation_log_main_path is not None:
        raise ValueError(
            "simulation_log_main_path must be null when running simulation."
        )
    run_simulation(cfg=cfg)


if __name__ == "__main__":
    main()
