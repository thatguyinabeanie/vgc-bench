import asyncio
import os

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from agent import Agent
from env import ShowdownEnv

BATTLE_FORMAT = "gen4randombattle"


async def train():
    # setup
    ppo = PPO(
        "MlpPolicy",
        ShowdownEnv(RandomPlayer()),
        learning_rate=lr_schedule,
        tensorboard_log="output/logs/ppo",
    )
    if os.path.exists("output/saves/ppo"):
        ppo = ppo.load("output/saves/ppo")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40)
    ppo.set_env(env)
    # train
    ppo = ppo.learn(1_000_000, reset_num_timesteps=False, progress_bar=True)
    # evaluate
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT)
    random = RandomPlayer(battle_format=BATTLE_FORMAT)
    max_damage = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT)
    simple_heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT)
    result = await cross_evaluate([agent, random, max_damage, simple_heuristic], n_challenges=10)
    print(result[agent.username])
    ppo.save("output/saves/ppo")


def lr_schedule(progress: float) -> float:
    return 10 ** (-4.23) / (8 * progress + 1) ** 1.5


if __name__ == "__main__":
    asyncio.run(train())
