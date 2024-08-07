import os
import random
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from policy import MaskedActorCriticPolicy

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self, num_saved_timesteps: int, save_interval: int):
        super().__init__()
        self.num_saved_timesteps = num_saved_timesteps
        self.save_interval = save_interval
        self.policy_pool = [
            PPO.load(f"saves/{filename}").policy for filename in os.listdir("saves")
        ]

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = [MaskedActorCriticPolicy.clone(self.model)] + self.policy_pool
        for i in range(self.model.env.num_envs):
            policy = random.choice(policies)
            self.model.env.env_method("set_opp_policy", policy, indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_interval == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.policy_pool.append(new_policy)
            self.model.save(f"saves/{self.model.num_timesteps}")
