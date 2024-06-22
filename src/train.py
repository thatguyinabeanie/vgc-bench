import asyncio
from copy import deepcopy

from gymnasium.utils.env_checker import check_env
from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from agent import Agent
from environment import ShowdownEnv
from nn import MLP


async def train():
    agent = Agent(MLP(1404, [100, 100], 10), battle_format="gen4randombattle")
    random = RandomPlayer(battle_format="gen4randombattle")
    max_damage = MaxBasePowerPlayer(battle_format="gen4randombattle")
    simple_heuristic = SimpleHeuristicsPlayer(battle_format="gen4randombattle")
    while True:
        opponent = Agent(deepcopy(agent.nn), battle_format="gen4randombattle")
        env = ShowdownEnv(opponent, battle_format="gen4randombattle")
        check_env(env, skip_render_check=True)
        ppo = PPO("MlpPolicy", env)
        ppo = ppo.learn(100)
        result = await cross_evaluate(
            [agent, random, max_damage, simple_heuristic], n_challenges=10
        )
        print(result[agent.username])


if __name__ == "__main__":
    asyncio.run(train())
