import asyncio
import os
import time
from subprocess import DEVNULL, Popen

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from agent import Agent
from teams import RandomTeamBuilder


def run_eval():
    Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    time.sleep(5)
    num_saved_timesteps = 0
    if os.path.exists("saves") and len(os.listdir("saves")) > 0:
        files = os.listdir("saves")
        num_saved_timesteps = max([int(file[:-4]) for file in files])
    policy = PPO.load(f"saves/{num_saved_timesteps}.zip").policy
    eval_agent = Agent(
        policy,
        account_configuration=AccountConfiguration("Agent", None),
        battle_format="gen9ou",
        log_level=40,
        max_concurrent_battles=10,
        team=RandomTeamBuilder(),
    )
    eval_opponents: list[Player] = [
        RandomPlayer(
            account_configuration=AccountConfiguration("Random", None),
            battle_format="gen9ou",
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        ),
        MaxBasePowerPlayer(
            account_configuration=AccountConfiguration("Power", None),
            battle_format="gen9ou",
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        ),
        SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("Heuristics", None),
            battle_format="gen9ou",
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        ),
    ]
    result = asyncio.run(eval_agent.battle_against_multi(eval_opponents, n_battles=100))
    print(result)


if __name__ == "__main__":
    run_eval()
