import argparse
import asyncio
import json
import os
from subprocess import DEVNULL, Popen

import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from stable_baselines3 import PPO

from agent import Agent
from teams import RandomTeamBuilder


async def play(n_games: int, play_on_ladder: bool):
    print("Setting up...")
    process = Popen(
        ["node", "pokemon-showdown", "start", "8001"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    await asyncio.sleep(5)
    with open("logs/win_rates.json") as f:
        win_rates = json.load(f)
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        i = np.argmax(win_rates)
        policy = PPO.load(f"saves/{files[i][:-4]}").policy
        print(f"Loaded {files[i]}")
    else:
        raise FileNotFoundError()
    agent = Agent(
        policy,
        account_configuration=AccountConfiguration("", ""),  # fill in
        battle_format="gen9ou",
        log_level=40,
        max_concurrent_battles=10,
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=play_on_ladder,
        team=RandomTeamBuilder(),
    )
    process.terminate()
    process.wait()
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
