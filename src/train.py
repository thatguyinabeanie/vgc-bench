import asyncio
from copy import deepcopy

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from agent import Agent
from nn import MLP


async def train():
    nn = MLP(1404, [100, 100], 10)
    agent = Agent(nn, battle_format="gen4randombattle")
    opponent = Agent(deepcopy(nn), battle_format="gen4randombattle")
    random = RandomPlayer(battle_format="gen4randombattle")
    max_damage = MaxBasePowerPlayer(battle_format="gen4randombattle")
    simple_heuristic = SimpleHeuristicsPlayer(battle_format="gen4randombattle")
    while True:
        # self-play
        await cross_evaluate([agent, opponent], n_challenges=100)
        # training
        for _ in agent.experiences + opponent.experiences:
            # TODO: implement training
            pass
        # evaluation
        result = await cross_evaluate(
            [agent, random, max_damage, simple_heuristic], n_challenges=10
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(train())
