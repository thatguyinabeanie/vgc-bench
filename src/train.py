import asyncio

from poke_env import cross_evaluate
from poke_env.player import RandomPlayer

from basic_player import BasicPlayer


async def train():
    player = BasicPlayer()
    opponent = RandomPlayer()
    result = await cross_evaluate([player, opponent], n_challenges=10)
    print(result)


if __name__ == "__main__":
    asyncio.run(train())
