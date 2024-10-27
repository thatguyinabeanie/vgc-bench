import argparse
import asyncio
import json
import os

import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from stable_baselines3 import PPO

from agent import Agent
from teams import RandomTeamBuilder


async def play(teams: list[int], opp_teams: list[int], n_games: int, play_on_ladder: bool):
    print("Setting up...")
    run_name = f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"
    if os.path.exists(f"saves-teampreview-random-again/{run_name}") and len(os.listdir(f"saves-teampreview-random-again/{run_name}")) > 0:
        files = os.listdir(f"saves-teampreview-random-again/{run_name}")
        with open(f"logs/{run_name}-win_rates.json") as f:
            win_rates = json.load(f)
        i = np.argmax(win_rates)
        policy = PPO.load(f"saves/{run_name}/{files[i][:-4]}").policy
        print(f"Loaded {files[i]}.")
    else:
        raise FileNotFoundError()
    agent = Agent(
        policy,
        account_configuration=AccountConfiguration("", ""),  # fill in
        battle_format="gen9vgc2024regh",
        log_level=40,
        max_concurrent_battles=10,
        server_configuration=ShowdownServerConfiguration,
        accept_open_team_sheet=True,
        start_timer_on_battle_start=play_on_ladder,
        team=RandomTeamBuilder(teams, "gen9vgc2024regh"),
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
    parser.add_argument("--teams", nargs="+", type=int)
    parser.add_argument("--opp_teams", nargs="+", type=int)
    parser.add_argument("-n", type=int, default=1, help="Number of games to play. Default is 1.")
    parser.add_argument("-l", action="store_true", help="Play ladder. Default accepts challenges.")
    args = parser.parse_args()
    asyncio.run(play(args.teams, args.opp_teams, args.n, args.l))
