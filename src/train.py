import os
import time
from subprocess import DEVNULL, Popen

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import Callback
from env import ShowdownEnv
from policy import MaskedActorCriticPolicy


def train():
    Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    time.sleep(5)
    battle_format = "gen9ou"
    num_saved_timesteps = 0
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        num_saved_timesteps = max([int(file[:-3]) for file in files])
    env = SubprocVecEnv([lambda i=i: ShowdownEnv.create_env(i, battle_format) for i in range(16)])
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=lambda x: 1e-4 / (8 * (1 - x) + 1) ** 1.5,
        n_steps=512,
        batch_size=1024,
        tensorboard_log="logs",
    )
    if num_saved_timesteps > 0:
        ppo.policy = torch.load(f"saves/{num_saved_timesteps}.pt")
    callback = Callback(
        num_saved_timesteps, battle_format, save_interval=102_400, bench_interval=102_400
    )
    ppo = ppo.learn(100_000_000 - num_saved_timesteps, callback=callback, reset_num_timesteps=False)


if __name__ == "__main__":
    train()
