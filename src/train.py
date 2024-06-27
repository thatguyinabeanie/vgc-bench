import asyncio
import os
import subprocess

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


class SaveAndReplaceOpponentCallback(BaseCallback):
    def __init__(self, save_freq: int):
        super().__init__()
        self.save_freq = save_freq
        self.rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        opponent = Agent(self.model.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        self.model.env.set_opponent(opponent)  # type: ignore
        self.rollout_count += 1
        if self.rollout_count % self.save_freq == 0:
            self.model.save(f"output/saves/ppo_{1024 * self.rollout_count}")


async def train():
    # start showdown server
    subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        stdout=subprocess.DEVNULL,
        cwd="pokemon-showdown",
    )
    await asyncio.sleep(5)
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    ppo = PPO(
        "MlpPolicy",
        wrapper_env,
        learning_rate=calc_learning_rate,
        n_steps=1024,
        tensorboard_log="output/logs/ppo",
    )
    if os.path.exists("output/saves/ppo.zip"):
        ppo.set_parameters("output/saves/ppo.zip")
        print("Resuming old run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    ppo.env.set_opponent(opponent)  # type: ignore
    # train
    callback = SaveAndReplaceOpponentCallback(save_freq=100)
    ppo = ppo.learn(100_000_000, callback=callback, reset_num_timesteps=False)
    # evaluate
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    random = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    max_power = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    results = await agent.battle_against_multi([random, max_power, heuristic], n_battles=100)
    print(results)


def calc_learning_rate(progress_remaining: float) -> float:
    progress = 1 - progress_remaining
    current_progress = 0
    return 10**-4.23 / (8 * ((progress + current_progress) / (1 + current_progress)) + 1) ** 1.5


if __name__ == "__main__":
    asyncio.run(train())
