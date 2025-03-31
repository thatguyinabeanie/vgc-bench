import argparse

import torch
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from src.agent import Agent
from src.callback import Callback
from src.teams import RandomTeamBuilder
from src.utils import battle_format, num_frames
from stable_baselines3 import PPO


def eval(filepath: str, num_teams: int, port: int):
    eval_agent = Agent(
        PPO.load("results/saves-bc-fs/20.zip").policy,
        device=torch.device("cuda:0"),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    eval_opponent = Agent(
        PPO.load("results/saves-bc-fs/15.zip").policy,
        device=torch.device("cuda:0"),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    win_rate = Callback.compare(eval_agent, eval_opponent, n_battles=100)
    print(win_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Pokémon AI model")
    parser.add_argument("--filepath", type=str, help="Path of model to evaluate")
    parser.add_argument("--num_teams", type=int, default=1, help="Number of teams to eval with")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    args = parser.parse_args()
    eval(args.filepath, args.num_teams, args.port)
