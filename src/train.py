import asyncio
import os

from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
            opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
            ppo.env.set_opponent(opponent)  # type: ignore
        return True


async def train():
    # setup
    global ppo
    if os.path.exists("output/saves/ppo.zip"):
        ppo.set_parameters("output/saves/ppo.zip")
        print("Resuming old run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    vec_env = DummyVecEnv([lambda: env])
    custom_env = ShowdownVecEnvWrapper(vec_env, env)
    ppo.set_env(custom_env)
    # train
    callback = SelfPlayCallback(check_freq=10_000)
    ppo = ppo.learn(100_000, callback=callback, reset_num_timesteps=False)
    # evaluate
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    random = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    max_power = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    heuristics = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    result = await cross_evaluate([agent, random, max_power, heuristics], n_challenges=100)
    print(result[agent.username])
    ppo.save("output/saves/ppo")


if __name__ == "__main__":
    asyncio.run(train())
