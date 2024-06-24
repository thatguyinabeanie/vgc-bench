import asyncio
import os
import time

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from agent import Agent
from env import ShowdownEnv

BATTLE_FORMAT = "gen4randombattle"


async def train():
    dummy_env = ShowdownEnv(RandomPlayer())
    ppo = PPO("MlpPolicy", dummy_env, n_epochs=100, tensorboard_log="PPO")
    if os.path.exists("ppo"):
        ppo = ppo.load("ppo")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40)
    ppo.set_env(env)
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT)
    random = RandomPlayer(battle_format=BATTLE_FORMAT)
    max_damage = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT)
    simple_heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT)
    while True:
        ppo = ppo.learn(10_000, reset_num_timesteps=False, progress_bar=True)
        agent.policy = ppo.policy
        result = await cross_evaluate(
            [agent, random, max_damage, simple_heuristic], n_challenges=10
        )
        print(time.strftime("%H:%M:%S"), "-", result[agent.username])
        ppo.save("ppo")


if __name__ == "__main__":
    asyncio.run(train())
