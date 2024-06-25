import asyncio
import os

from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO

from env import ShowdownEnv

BATTLE_FORMAT = "gen4ou"
TEAM1 = """
Bronzong @ Lum Berry
Ability: Heatproof
EVs: 248 HP / 252 Atk / 4 Def / 4 SpD
Brave Nature
IVs: 0 Spe
- Stealth Rock
- Gyro Ball
- Earthquake
- Explosion

Gengar @ Life Orb
Ability: Levitate
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Shadow Ball
- Focus Blast
- Explosion
- Hidden Power [Fire]

Tyranitar (M) @ Choice Band
Ability: Sand Stream
EVs: 252 Atk / 44 SpD / 212 Spe
Adamant Nature
- Stone Edge
- Crunch
- Pursuit
- Superpower

Kingdra @ Choice Specs
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hydro Pump
- Draco Meteor
- Surf
- Dragon Pulse

Flygon (M) @ Choice Scarf
Ability: Levitate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- U-turn
- Outrage
- Earthquake
- Thunder Punch

Lucario @ Life Orb
Ability: Inner Focus
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Close Combat
- Swords Dance
- Bullet Punch
- Extreme Speed"""
TEAM2 = """
Bronzong @ Lum Berry
Ability: Heatproof
EVs: 248 HP / 252 Atk / 8 SpD
Brave Nature
IVs: 0 Spe
- Gyro Ball
- Stealth Rock
- Earthquake
- Explosion

Dragonite (M) @ Choice Band
Ability: Inner Focus
EVs: 48 HP / 252 Atk / 208 Spe
Adamant Nature
- Outrage
- Dragon Claw
- Extreme Speed
- Earthquake

Mamoswine (M) @ Life Orb
Ability: Oblivious
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Ice Shard
- Earthquake
- Stone Edge
- Superpower

Magnezone @ Leftovers
Ability: Magnet Pull
EVs: 140 HP / 252 SpA / 116 Spe
Modest Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Thunderbolt
- Thunder Wave
- Substitute
- Hidden Power [Fire]

Flygon @ Choice Scarf
Ability: Levitate
EVs: 252 Atk / 6 Def / 252 Spe
Adamant Nature
- Outrage
- Earthquake
- Stone Edge
- U-turn

Kingdra (M) @ Chesto Berry
Ability: Swift Swim
EVs: 144 HP / 160 Atk / 40 SpD / 164 Spe
Adamant Nature
- Dragon Dance
- Waterfall
- Outrage
- Rest"""


async def train():
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM2)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM1)
    ppo = PPO("MlpPolicy", env, learning_rate=lr_schedule, tensorboard_log="output/logs/ppo")
    if os.path.exists("output/saves/ppo"):
        ppo = ppo.load("output/saves/ppo")
    ppo = ppo.learn(1_000_000, progress_bar=True)
    ppo.save("output/saves/ppo")


def lr_schedule(progress: float) -> float:
    return 10 ** (-4.23) / (8 * progress + 1) ** 1.5


if __name__ == "__main__":
    asyncio.run(train())
