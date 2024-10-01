import argparse
import asyncio
import json
import os

import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from stable_baselines3 import PPO

from agent import Agent
from teams import RandomTeamBuilder


async def play(n_games: int, play_on_ladder: bool, n_teams: int = 1):
    print("Setting up...")
    if os.path.exists(f"saves/{n_teams}_teams") and len(os.listdir(f"saves/{n_teams}_teams")) > 0:
        files = os.listdir(f"saves/{n_teams}_teams")
        with open(f"logs/{n_teams}_teams_win_rates.json") as f:
            win_rates = json.load(f)
        i = np.argmax(win_rates)
        policy = PPO.load(f"saves/{n_teams}_teams/{files[i][:-4]}").policy
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
        start_timer_on_battle_start=play_on_ladder,
        team=RandomTeamBuilder(1, "gen9vgc2024regh"),
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
    parser.add_argument("-n", type=int, default=1, help="Number of games to play. Default is 1.")
    parser.add_argument("-l", action="store_true", help="Play ladder. Default accepts challenges.")
    args = parser.parse_args()
    asyncio.run(play(args.n, args.l))
