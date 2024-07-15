import argparse
import asyncio
import os
from subprocess import DEVNULL, Popen

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from agent import Agent
from env import ShowdownEnv
from policy import MaskedActorCriticPolicy
from teams import TEAM1


async def play(play_on_ladder: bool, n_games: int):
    print("Setting up...")
    process = Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        cwd="pokemon-showdown",
    )
    await asyncio.sleep(5)
    model = PPO(MaskedActorCriticPolicy, ShowdownEnv(RandomPlayer()))
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])
        model.set_parameters(os.path.join("output/saves", f"ppo_{num_saved_timesteps}.zip"))
    agent = Agent(
        MaskedActorCriticPolicy.clone(model),
        account_configuration=AccountConfiguration("", ""),  # fill in
        battle_format="gen9ou",
        log_level=40,
        max_concurrent_battles=10,
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=play_on_ladder,
        team=TEAM1,
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
