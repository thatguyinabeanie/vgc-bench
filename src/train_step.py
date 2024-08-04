import logging
import os

import torch
from poke_env import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import Agent
from callback import Callback
from env import ShowdownEnv
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


def train_step():
    logging.basicConfig(
        filename="debug.log",
        filemode="w",
        format="%(name)s %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s()\n%(message)s\n",
        level=logging.INFO,
    )
    total_steps = 10_000_000
    steps = 102_400
    num_envs = 16
    battle_format = "gen9ou"

    def create_env(i: int):
        num_gpus = torch.cuda.device_count()
        return ShowdownEnv(
            Agent(
                None,
                device=torch.device(f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu"),
                account_configuration=AccountConfiguration(f"Opponent{i + 1}", None),
                battle_format=battle_format,
                log_level=40,
                team=RandomTeamBuilder(),
            ),
            account_configuration=AccountConfiguration(f"Agent{i + 1}", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )

    env = SubprocVecEnv([lambda i=i: create_env(i) for i in range(num_envs)], start_method="spawn")
    num_saved_timesteps = 0
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = (1 - progress_remaining) * (num_saved_timesteps + steps) / total_steps
        return 1e-4 / (8 * progress + 1) ** 1.5

    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=calc_learning_rate,
        n_steps=256,
        batch_size=1024,
        tensorboard_log="logs",
    )
    if num_saved_timesteps > 0:
        ppo.set_parameters(os.path.join("saves", f"ppo_{num_saved_timesteps}.zip"))
    callback = Callback(num_envs, num_saved_timesteps, steps, battle_format)
    ppo = ppo.learn(num_saved_timesteps + steps, callback=callback, reset_num_timesteps=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_step()
