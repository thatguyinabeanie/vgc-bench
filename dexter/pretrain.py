import argparse
import os
import pickle
import warnings

import numpy as np
from imitation.algorithms.bc import BC
from src.env import ShowdownDoublesEnv, ShowdownSinglesEnv
from src.policy import MaskedActorCriticPolicy
from stable_baselines3 import PPO

warnings.filterwarnings("ignore", category=UserWarning)


def pretrain(teams: list[int], opp_teams: list[int], port: int, device: str):
    run_name = f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"
    if os.path.exists(f"saves/{run_name}/0.zip"):
        print("pretrained model already exists - skipping pretraining", flush=True)
        return
    battle_format = "gen9vgc2024regh"
    num_frames = 3
    self_play = True
    env_class = ShowdownDoublesEnv if "vgc" in battle_format else ShowdownSinglesEnv
    env = env_class.create_env(
        0,
        battle_format,
        num_frames,
        port,
        teams,
        opp_teams,
        self_play,
        device,
        start_listening=False,
    )
    ppo = PPO(MaskedActorCriticPolicy, env, policy_kwargs={"num_frames": num_frames}, device=device)
    with open("data/trajs.pkl", "rb") as f:
        trajs = pickle.load(f)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", nargs="+", type=int, required=False)
    parser.add_argument("--opp_teams", nargs="+", type=int, required=False)
    parser.add_argument("--num_teams", type=int, required=False)
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    assert args.num_teams or (args.teams and args.opp_teams)
    teams: list[int] = args.teams or list(range(args.num_teams))
    opp_teams: list[int] = args.opp_teams or list(range(args.num_teams))
    pretrain(teams, opp_teams, args.port, args.device)
