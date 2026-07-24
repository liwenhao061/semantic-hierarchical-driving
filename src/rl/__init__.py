"""
RL closed-loop package for PLUTO.

Modules
-------
nuplan_env_wrapper      : gym-style environment wrapping nuplan scenario data
rollout_buffer          : PPO rollout buffer with GAE for reward + cost
ppo_updater             : PPO actor-critic parameter update
cmdp_dual_update        : CMDP Lagrangian dual-variable update + shielding
teacher_rollout_collector: collect teacher trajectories for distillation
rl_trainer              : orchestrates env interaction → buffer → PPO → CMDP
"""

from .nuplan_env_wrapper import NuplanEnvWrapper
from .rollout_buffer import RolloutBuffer
from .ppo_updater import PPOUpdater
from .cmdp_dual_update import CMDPDualUpdater
from .teacher_rollout_collector import TeacherRolloutCollector
from .rl_trainer import RLTrainer

__all__ = [
    "NuplanEnvWrapper",
    "RolloutBuffer",
    "PPOUpdater",
    "CMDPDualUpdater",
    "TeacherRolloutCollector",
    "RLTrainer",
]
