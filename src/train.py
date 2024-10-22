import argparse
import json
import os
import time
from subprocess import PIPE, STDOUT, Popen

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import Agent
from callback import Callback
from env import ShowdownDoublesEnv, ShowdownSinglesEnv
from policy import MaskedActorCriticPolicy


def train(teams: list[int], opp_teams: list[int], port: int, device: str):
    server = Popen(
        ["node", "pokemon-showdown", "start", str(port), "--no-security"],
        stdout=PIPE,
        stderr=STDOUT,
        cwd="pokemon-showdown",
    )
    time.sleep(10)
    steps = 98_304
    num_envs = 32
    battle_format = "gen9vgc2024regh"
    self_play = False
    env_class = ShowdownDoublesEnv if "vgc" in battle_format else ShowdownSinglesEnv
    env = SubprocVecEnv(
        [
            lambda i=i: Monitor(
                env_class.create_env(i, battle_format, port, teams, opp_teams, self_play, device)
            )
            for i in range(num_envs)
        ]
    )
    run_name = f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"
    num_saved_timesteps = 0
    if os.path.exists(f"saves/{run_name}") and len(os.listdir(f"saves/{run_name}")) > 0:
        files = os.listdir(f"saves/{run_name}")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=2048 // num_envs,
        batch_size=64,
        gamma=1,
        ent_coef=0.1 * 0.75 ** (num_saved_timesteps // steps),
        tensorboard_log="logs",
        policy_kwargs={
            "mask_len": (
                2 * Agent.doubles_act_len if "vgc" in battle_format else Agent.singles_act_len
            )
        },
        device=device,
    )
    if num_saved_timesteps > 0:
        ppo.set_parameters(f"saves/{run_name}/{num_saved_timesteps}.zip", device=ppo.device)
        ppo.num_timesteps = num_saved_timesteps
    else:
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open(f"logs/{run_name}-win_rates.json", "w") as f:
            json.dump([], f)
    callback = Callback(steps, battle_format, teams, opp_teams, port, self_play)
    ppo.learn(steps, callback=callback, tb_log_name=run_name, reset_num_timesteps=False)
    server.terminate()
    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", nargs="+", type=int)
    parser.add_argument("--opp_teams", nargs="+", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    while True:
        train(args.teams, args.opp_teams, args.port, args.device)
