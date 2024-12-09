import os
import pickle
import time
from subprocess import PIPE, STDOUT, Popen

import numpy as np
from imitation.algorithms.bc import BC
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from src.callback import Callback
from src.env import ShowdownDoublesEnv, ShowdownSinglesEnv
from src.policy import MaskedActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

battle_format = "gen9vgc2024regh"
device = "cuda:3"
num_envs = 32
num_frames = 3
num_teams = 16
port = 8000
self_play = True
steps = 98_304

env_class = ShowdownDoublesEnv if "vgc" in battle_format else ShowdownSinglesEnv
opp_teams = list(range(num_teams))
teams = list(range(num_teams))
run_name = f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"


def pretrain():
    env = env_class(
        RandomPlayer(
            account_configuration=AccountConfiguration("DummyPlayer", None),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            start_listening=False,
        ),
        account_configuration=AccountConfiguration(f"DummyEnv", None),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    ppo = PPO(MaskedActorCriticPolicy, env, policy_kwargs={"num_frames": num_frames}, device=device)
    print("loading trajectories...", flush=True)
    with open("data/trajs.pkl", "rb") as f:
        trajs = pickle.load(f)
    print(f"loaded {len(trajs)} trajectories", flush=True)
    bc = BC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=trajs,
        device=device,
    )
    bc.train(n_epochs=10)
    ppo.save(f"saves/{run_name}/0")


def train():
    server = Popen(
        ["node", "pokemon-showdown", "start", str(port), "--no-security"],
        stdout=PIPE,
        stderr=STDOUT,
        cwd="pokemon-showdown",
    )
    time.sleep(10)
    env = SubprocVecEnv(
        [
            lambda i=i: Monitor(
                env_class.create_env(
                    i, battle_format, num_frames, port, teams, opp_teams, self_play, device
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
        ent_coef=0.02,
        tensorboard_log="logs",
        policy_kwargs={"num_frames": num_frames},
        device=device,
    )
    num_saved_timesteps = max([int(file[:-4]) for file in os.listdir(f"saves/{run_name}")])
    ppo.num_timesteps = num_saved_timesteps
    ppo.set_parameters(f"saves/{run_name}/{num_saved_timesteps}.zip", device=ppo.device)
    callback = Callback(steps, battle_format, num_frames, teams, opp_teams, port, self_play)
    ppo.learn(steps, callback=callback, tb_log_name=run_name, reset_num_timesteps=False)
    server.terminate()
    server.wait()


if __name__ == "__main__":
    if not os.path.exists(f"saves/{run_name}/0.zip"):
        pretrain()
    while True:
        train()
