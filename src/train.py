import asyncio

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from agent import Agent

sample_team = """
Gengar
- Hypnosis
- Thunderbolt
- Night Shade
- Explosion

Chansey
- Ice Beam
- Thunderbolt
- Thunder Wave
- Soft-Boiled

Snorlax
- Body Slam
- Earthquake
- Hyper Beam
- Self-Destruct

Exeggutor
- Sleep Powder
- Stun Spore
- Psychic
- Explosion

Tauros
- Body Slam
- Earthquake
- Hyper Beam
- Blizzard

Lapras
- Blizzard
- Thunderbolt
- Body Slam
- Confuse Ray
"""


async def train():
    agent = Agent(battle_format="gen1ou", team=sample_team)
    random = RandomPlayer(battle_format="gen1ou", team=sample_team)
    max_damage = MaxBasePowerPlayer(battle_format="gen1ou", team=sample_team)
    simple_heuristic = SimpleHeuristicsPlayer(battle_format="gen1ou", team=sample_team)
    result = await cross_evaluate([agent, random, max_damage, simple_heuristic], n_challenges=10)
    print(result)


if __name__ == "__main__":
    asyncio.run(train())
