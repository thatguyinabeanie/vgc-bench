import json
import os
import time
from subprocess import DEVNULL, Popen

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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
    total_timesteps = 983_040_000
    num_envs = 32
    battle_format = "gen9ou"
    self_play = False
    env = SubprocVecEnv(
        [
            lambda i=i: Monitor(ShowdownEnv.create_env(i, battle_format, self_play))
            for i in range(num_envs)
        ]
    )
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=2048 // num_envs,
        batch_size=64,
        # rollout_buffer_class=MultiAgentBuffer,
        # rollout_buffer_kwargs={"buffer_size_override": 2048 // num_envs * 2},
        tensorboard_log="logs",
        device="cuda:0",
    )
    num_saved_timesteps = 0
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
        ppo.set_parameters(f"saves/{num_saved_timesteps}.zip")
    if num_saved_timesteps == 0:
        with open("logs/win_rates.json", "w") as f:
            json.dump([], f)
    ppo.num_timesteps = num_saved_timesteps
    callback = Callback(98_304, battle_format, self_play)
    ppo = ppo.learn(total_timesteps, callback=callback, reset_num_timesteps=False)


if __name__ == "__main__":
    train()
