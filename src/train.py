import os
import time
from subprocess import DEVNULL, Popen

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
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
    num_envs = 32
    battle_format = "gen9ou"
    env = SubprocVecEnv(
        [lambda i=i: ShowdownEnv.create_env(i, battle_format) for i in range(num_envs)]
    )
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=lambda x: 1e-4 / (8 * (1 - x) + 1) ** 1.5,
        n_steps=2048 // num_envs,
        device="cuda:0",
    )
    logger = configure("logs", ["json", "tensorboard"])
    ppo.set_logger(logger)
    num_saved_timesteps = 0
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
        ppo.set_parameters(f"saves/{num_saved_timesteps}.zip")
    ppo.num_timesteps = num_saved_timesteps
    callback = Callback(save_interval=102_400, battle_format=battle_format)
    ppo = ppo.learn(10_000_000, callback=callback, reset_num_timesteps=False)


if __name__ == "__main__":
    train()
