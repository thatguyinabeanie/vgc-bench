import asyncio

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from agent import Agent


async def train():
    agent = Agent(battle_format="gen4randombattle")
    random = RandomPlayer(battle_format="gen4randombattle")
    max_damage = MaxBasePowerPlayer(battle_format="gen4randombattle")
    simple_heuristic = SimpleHeuristicsPlayer(battle_format="gen4randombattle")
    result = await cross_evaluate([agent, random, max_damage, simple_heuristic], n_challenges=10)
    print(result)


if __name__ == "__main__":
    asyncio.run(train())
