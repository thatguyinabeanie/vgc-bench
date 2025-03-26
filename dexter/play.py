import argparse
import asyncio
import json
import os

import numpy as np
from nashpy import Game
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from src.agent import Agent
from src.teams import RandomTeamBuilder
from src.utils import battle_format, double_oracle, num_frames
from stable_baselines3 import PPO


async def play(num_teams: int, n_games: int, play_on_ladder: bool):
    print("Setting up...")
    if (
        os.path.exists(
            f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams"
        )
        and len(
            os.listdir(
                f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams"
            )
        )
        > 0
    ):
        files = os.listdir(
            f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams"
        )
        if double_oracle:
            with open(
                f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams-payoff-matrix.json"
            ) as f:
                payoff_matrix = np.array(json.load(f))
            g = Game(payoff_matrix)
            prob_dist = g.lemke_howson(0)[0]
            i = np.argmax(prob_dist)
        else:
            i = len(files) - 1
        policy = PPO.load(
            f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams/{files[i][:-4]}"
        ).policy
        print(f"Loaded {files[i]}.")
    else:
        raise FileNotFoundError()
    agent = Agent(
        policy,
        num_frames=num_frames,
        account_configuration=AccountConfiguration("", ""),  # fill in
        battle_format=battle_format,
        log_level=40,
        max_concurrent_battles=10,
        server_configuration=ShowdownServerConfiguration,
        accept_open_team_sheet=True,
        start_timer_on_battle_start=play_on_ladder,
        team=RandomTeamBuilder(list(range(num_teams)), battle_format),
    )
    if play_on_ladder:
        print("Entering ladder")
        await agent.ladder(n_games=n_games)
        print(f"{agent.n_won_battles}-{agent.n_lost_battles}-{agent.n_tied_battles}")
    else:
        print("AI is ready")
        await agent.accept_challenges(opponent=None, n_challenges=n_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_teams", type=int, default=1, help="Number of teams to train with")
    parser.add_argument("-n", type=int, default=1, help="Number of games to play. Default is 1.")
    parser.add_argument("-l", action="store_true", help="Play ladder. Default accepts challenges.")
    args = parser.parse_args()
    asyncio.run(play(args.num_teams, args.n, args.l))
