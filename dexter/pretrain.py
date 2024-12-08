import argparse
import asyncio
import json

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from src.env import ShowdownDoublesEnv, ShowdownSinglesEnv
from src.policy import MaskedActorCriticPolicy
from src.reader import LogReader
from stable_baselines3 import PPO


def pretrain(teams: list[int], opp_teams: list[int], port: int, device: str):
    num_envs = 1
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
    with open("json/human.json", "r") as f:
        logs = json.load(f)
    trajs = process_logs(logs, 100)
    bc = BC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=trajs,
    )
    bc.train(n_epochs=10)
    run_name = f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"
    ppo.save(f"saves/{run_name}/0")


def process_logs(log_jsons: dict[str, tuple[str, str]], n: int) -> list[Trajectory]:
    trajs = []
    for i, (tag, (_, log)) in enumerate(log_jsons.items()):
        if i == n:
            break
        print(
            f"conversion progress: {round(100 * i / min(n, len(log_jsons)), ndigits=2)}%",
            end="\r",
            flush=True,
        )
        player1 = LogReader(battle_format="gen9vgc2024regh", accept_open_team_sheet=True)
        player2 = LogReader(battle_format="gen9vgc2024regh", accept_open_team_sheet=True)
        obs1, logits1 = asyncio.run(player1.follow_log(tag, log, "p1"))
        obs2, logits2 = asyncio.run(player2.follow_log(tag, log, "p2"))
        trajs += [
            Trajectory(obs=obs1, acts=logits1, infos=None, terminal=True),
            Trajectory(obs=obs2, acts=logits2, infos=None, terminal=True),
        ]
    print("done!", flush=True)
    return trajs


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
