import argparse
import asyncio
import json
import os
import warnings

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from src.env import ShowdownDoublesEnv, ShowdownSinglesEnv
from src.policy import MaskedActorCriticPolicy
from src.reader import LogReader
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
    with open("json/human.json", "r") as f:
        logs = json.load(f)
    trajs = process_logs(logs)
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


def process_logs(log_jsons: dict[str, tuple[str, str]]) -> list[Trajectory]:
    print("preparing data...", flush=True)
    trajs = []
    total = 0
    for i, (tag, (_, log)) in enumerate(log_jsons.items()):
        print(f"{round(100 * i / len(log_jsons), ndigits=2)}%", end="\r", flush=True)
        player1 = LogReader(battle_format="gen9vgc2024regh", accept_open_team_sheet=True)
        player2 = LogReader(battle_format="gen9vgc2024regh", accept_open_team_sheet=True)
        result1 = asyncio.run(player1.follow_log(tag, log, "p1"))
        result2 = asyncio.run(player2.follow_log(tag, log, "p2"))
        if result1 is not None:
            states1, actions1 = result1
            total += len(states1)
            trajs += [Trajectory(obs=states1, acts=actions1, infos=None, terminal=True)]
        if result2 is not None:
            states2, actions2 = result2
            total += len(states2)
            trajs += [Trajectory(obs=states2, acts=actions2, infos=None, terminal=True)]
    print(f"prepared {len(trajs)} trajectories with {total} total state-action pairs", flush=True)
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
