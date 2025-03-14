import argparse
import asyncio

import torch
from poke_env.player import RandomPlayer
from poke_env.ps_client import ServerConfiguration
from src.agent import Agent
from src.teams import RandomTeamBuilder
from src.utils import battle_format, num_frames
from stable_baselines3 import PPO


def eval(filepath: str, port: int):
    eval_agent = Agent(
        PPO.load(filepath).policy,
        num_frames=num_frames,
        device=torch.device("cuda:0"),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(list(range(30)), battle_format),
    )
    eval_opponent = RandomPlayer(
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(list(range(30)), battle_format),
    )
    asyncio.run(eval_agent.battle_against(eval_opponent, n_battles=100))
    print(eval_agent.win_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Pokémon AI model")
    parser.add_argument("--filepath", type=str, help="Path of model to evaluate")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="CUDA device to use for eval",
    )
    args = parser.parse_args()
    eval(args.filepath, args.port)
