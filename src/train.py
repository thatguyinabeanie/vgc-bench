import asyncio
import os

from poke_env.player import RandomPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from env import ShowdownEnv, ShowdownVecEnvWrapper

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
ppo = PPO(
    "MlpPolicy", ShowdownEnv(RandomPlayer()), learning_rate=1e-5, tensorboard_log="output/logs/ppo"
)


class SelfPlayCallback(BaseCallback):
    def __init__(self, check_freq: int):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        global ppo
        if self.n_calls % self.check_freq == 0:
            opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM2)
            ppo.env.set_opponent(opponent)  # type: ignore
        return True


async def train():
    global ppo
    if os.path.exists("output/saves/ppo.zip"):
        ppo.set_parameters("output/saves/ppo.zip")
        print("Resuming old run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM2)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM1)
    vec_env = DummyVecEnv([lambda: env])
    custom_env = ShowdownVecEnvWrapper(vec_env, env)
    ppo.set_env(custom_env)
    callback = SelfPlayCallback(check_freq=10_000)
    ppo = ppo.learn(100_000, callback=callback, reset_num_timesteps=False)
    ppo.save("output/saves/ppo")


if __name__ == "__main__":
    asyncio.run(train())
