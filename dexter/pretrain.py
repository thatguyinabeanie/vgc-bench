import argparse
import pickle

import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from imitation.util.logger import configure
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer, SingleAgentWrapper
from poke_env.ps_client import ServerConfiguration
from src.agent import Agent
from src.callback import Callback
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import battle_format, frame_stack, num_frames
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
    trajs = load_trajs("data/trajs.pkl")
    if frame_stack:
        trajs = (frame_stack_traj(traj) for traj in trajs)
    bc = BC(
        observation_space=ppo.observation_space,  # type: ignore
        action_space=ppo.action_space,  # type: ignore
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=trajs,
        batch_size=1024,
        device=device,
        custom_logger=configure(f"results/logs-bc{'-fs' if frame_stack else ''}", ["tensorboard"]),
    )
    eval_agent = Agent(
        policy=None,
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
    eval_opponent = SimpleHeuristicsPlayer(
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    win_rate = Callback.compare(eval_agent, eval_opponent, 100)
    bc.logger.record("bc/eval", win_rate)
    ppo.save(f"results/saves-bc{'-fs' if frame_stack else ''}/0")
    for i in range(100):
        bc.train(n_epochs=10)
        policy = MaskedActorCriticPolicy.clone(ppo)
        eval_agent.set_policy(policy)
        win_rate = Callback.compare(eval_agent, eval_opponent, 100)
        bc.logger.record("bc/eval", win_rate)
        ppo.save(f"results/saves-bc{'-fs' if frame_stack else ''}/{i + 1}")


def load_trajs(filepath: str):
    with open(filepath, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def frame_stack_traj(traj: Trajectory) -> Trajectory:
    traj_len, *obs_shape = traj.obs.shape
    stacked_obs = np.empty((traj_len, num_frames, *obs_shape), dtype=traj.obs.dtype)
    zero_obs = np.zeros(obs_shape, dtype=traj.obs[0].dtype)
    for i in range(traj_len):
        for j in range(num_frames):
            idx = i - j
            if idx >= 0:
                stacked_obs[i, num_frames - 1 - j] = traj.obs[idx]
            else:
                stacked_obs[i, num_frames - 1 - j] = zero_obs
    return Trajectory(obs=stacked_obs, acts=traj.acts, infos=None, terminal=True)


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
    pretrain(args.num_teams, args.port, args.device)
