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


async def listen():
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
        battle_format="gen4ou",
        log_level=0,
        server_configuration=ShowdownServerConfiguration,
        team=TEAM1,
    )
    process.terminate()
    process.wait()
    print("AI is ready")
    await agent.accept_challenges(opponent=None, n_challenges=1000)


if __name__ == "__main__":
    asyncio.run(listen())
