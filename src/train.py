import asyncio
import os
import time

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from env import ShowdownEnv, ShowdownVecEnvWrapper

BATTLE_FORMAT = "gen4ou"
TEAM = """
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


async def train():
    # setup
    dummy_opponent = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(dummy_opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    vec_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    ppo = PPO("MlpPolicy", vec_env, learning_rate=1e-5, tensorboard_log="output/logs/ppo")
    if os.path.exists("output/saves/ppo.zip"):
        ppo.set_parameters("output/saves/ppo.zip")
        print("Resuming old run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    ppo.env.set_opponent(opponent)  # type: ignore
    random = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    max_power = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    heuristics = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    # repeatedly train, evaluate, save
    while True:
        ppo = ppo.learn(100_000, reset_num_timesteps=False)
        agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        opponent_copy = Agent(opponent.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        result = await cross_evaluate(
            [agent, opponent_copy, random, max_power, heuristics], n_challenges=100
        )
        print(time.strftime("%H:%M:%S"), "-", result[agent.username])
        num_wins = result[agent.username][opponent_copy.username]
        if num_wins is not None and num_wins >= 55:
            ppo.save("output/saves/ppo")
            ppo.env.set_opponent(agent)  # type: ignore
            print("Updating opponent.")


if __name__ == "__main__":
    asyncio.run(train())
