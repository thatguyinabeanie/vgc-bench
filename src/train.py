import argparse
import json
import os
import time
from subprocess import DEVNULL, Popen

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import Agent
from callback import Callback
from env import ShowdownDoublesEnv, ShowdownSinglesEnv
from policy import MaskedActorCriticPolicy


def train(num_teams: int, device: str):
    server = Popen(
        ["node", "pokemon-showdown", "start", str(8000 + num_teams), "--no-security"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    time.sleep(5)
    steps = 98_304
    num_envs = 32
    battle_format = "gen9vgc2024regh"
    self_play = False
    env_class = ShowdownDoublesEnv if "vgc" in battle_format else ShowdownSinglesEnv
    env = SubprocVecEnv(
        [
            lambda i=i: Monitor(
                env_class.create_env(
                    i, battle_format, 8000 + num_teams, num_teams, self_play, device
                )
            )
            for i in range(num_envs)
        ]
    )
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=2048 // num_envs,
        batch_size=64,
        gamma=1,
        tensorboard_log="logs",
        policy_kwargs={
            "mask_len": (
                2 * Agent.doubles_act_len if "vgc" in battle_format else Agent.singles_act_len
            )
        },
        device=device,
    )
    run_name = f"{num_teams}-teams"
    num_saved_timesteps = 0
    if os.path.exists(f"saves/{run_name}") and len(os.listdir(f"saves/{run_name}")) > 0:
        files = os.listdir(f"saves/{run_name}")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
        ppo.set_parameters(f"saves/{run_name}/{num_saved_timesteps}.zip", device=ppo.device)
    if num_saved_timesteps == 0:
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open(f"logs/{run_name}-win_rates.json", "w") as f:
            json.dump([], f)
    ppo.num_timesteps = num_saved_timesteps
    callback = Callback(steps, battle_format, num_teams, self_play)
    ppo.learn(steps, callback=callback, tb_log_name=run_name, reset_num_timesteps=False)
    server.terminate()
    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_teams", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    while True:
        train(args.num_teams, args.device)
