import argparse
import json
import os
import time
from subprocess import DEVNULL, Popen

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import Callback
from env import ShowdownEnv
from policy import MaskedActorCriticPolicy


def train(run_id: int):
    Popen(
        ["node", "pokemon-showdown", "start", str(8000 + run_id), "--no-security"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    time.sleep(5)
    total_timesteps = 983_040_000
    num_envs = 32
    battle_format = "gen9ou"
    num_teams = [1, 2, 4, 8]
    self_play = False
    env = SubprocVecEnv(
        [
            lambda i=i: Monitor(
                ShowdownEnv.create_env(
                    num_envs * run_id + i,
                    battle_format,
                    8000 + run_id,
                    num_teams[run_id],
                    self_play,
                )
            )
            for i in range(num_envs)
        ]
    )
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-4,
        n_steps=24_576 // num_envs,
        batch_size=1024,
        gamma=1,
        tensorboard_log=f"logs",
        device=f"cuda:{run_id % torch.cuda.device_count()}",
    )
    run_name = f"{num_teams[run_id]}-teams"
    num_saved_timesteps = 0
    if os.path.exists(f"saves/{run_name}") and len(os.listdir(f"saves/{run_name}")) > 0:
        files = os.listdir(f"saves/{run_name}")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
        ppo.set_parameters(f"saves/{run_name}/{num_saved_timesteps}.zip", device=ppo.device)
    if num_saved_timesteps == 0:
        if not os.path.exists(f"logs"):
            os.mkdir(f"logs")
        with open(f"logs/{run_name}-win_rates.json", "w") as f:
            json.dump([], f)
    ppo.num_timesteps = num_saved_timesteps
    callback = Callback(98_304, battle_format, run_id, num_teams[run_id], self_play)
    ppo.learn(total_timesteps, callback=callback, tb_log_name=run_name, reset_num_timesteps=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int)
    args = parser.parse_args()
    train(args.run_id)
