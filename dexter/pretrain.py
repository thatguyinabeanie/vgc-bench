import argparse
import asyncio
import os
import pickle

import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from imitation.util.logger import configure
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SingleAgentWrapper
from poke_env.ps_client import ServerConfiguration
from src.agent import Agent
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import battle_format, behavior_clone, doubles_chunk_obs_len, frame_stack, num_frames
from stable_baselines3 import PPO


def pretrain(num_teams: int, port: int, device: str):
    env = ShowdownEnv(
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    single_agent_env = SingleAgentWrapper(env, opponent)
    ppo = PPO(MaskedActorCriticPolicy, single_agent_env, device=device)
    with open("data/trajs.pkl", "rb") as f:
        trajs: list[Trajectory] = pickle.load(f)
    stacked_trajs = []
    if frame_stack:
        for i in range(len(trajs)):
            print(f"progress: {round(100 * i / len(trajs), ndigits=2)}%", end="\r")
            stacked_trajs += [frame_stack_traj(trajs[i])]
        trajs = stacked_trajs
    print(f"finished loading {len(trajs)} trajectories")
    bc = BC(
        observation_space=ppo.observation_space,  # type: ignore
        action_space=ppo.action_space,  # type: ignore
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=trajs,
        batch_size=1024,
        device=device,
        custom_logger=configure(f"logs/{num_teams}-teams", ["tensorboard"]),
    )
    eval_agent = Agent(
        policy=None,
        num_frames=num_frames,
        device=torch.device(device),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    eval_opponent = MaxBasePowerPlayer(
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    for i in range(10):
        bc.train(n_epochs=100)
        eval_agent.__policy = bc.policy
        asyncio.run(eval_agent.battle_against(eval_opponent, n_battles=100))
        bc.logger.record("bc/eval", eval_agent.win_rate)
        eval_agent.reset_battles()
        eval_opponent.reset_battles()
        ppo.save(f"saves/{num_teams}-teams/{i}")


def frame_stack_traj(traj: Trajectory) -> Trajectory:
    zero_obs = np.zeros([12, doubles_chunk_obs_len])
    obs_list = [
        np.stack([traj.obs[i - j] if i - j >= 0 else zero_obs for j in range(num_frames)], axis=0)
        for i in range(len(traj.obs))
    ]
    return Trajectory(obs=np.stack(obs_list, axis=0), acts=traj.acts, infos=None, terminal=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a Pokémon AI model")
    parser.add_argument("--num_teams", type=int, default=1, help="Number of teams to train with")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="CUDA device to use for training",
    )
    args = parser.parse_args()
    if not behavior_clone:
        print("behavior cloning toggled off - aborting")
    elif os.path.exists(f"saves/{args.num_teams}-teams/0.zip"):
        print("already have pretrained NN - aborting")
    else:
        pretrain(args.num_teams, args.port, args.device)
