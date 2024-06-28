import asyncio
import os

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
TOTAL_TIMESTEPS = 100_000_000


class SaveAndReplaceOpponentCallback(BaseCallback):
    def __init__(self, save_freq: int, num_saved_timesteps: int, n_steps: int):
        super().__init__()
        self.save_freq = save_freq
        self.total_timesteps = num_saved_timesteps
        self.n_steps = n_steps

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        global ppo
        opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        ppo.env.set_opponent(opponent)  # type: ignore
        self.total_timesteps += self.n_steps
        if self.total_timesteps % self.save_freq == 0:
            ppo.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")


async def train():
    # setup
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    global ppo
    ppo = PPO("MlpPolicy", wrapper_env, tensorboard_log="output/logs/ppo")
    num_saved_timesteps = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])
        ppo.set_parameters(f"output/saves/ppo_{num_saved_timesteps}.zip")
        print(f"Resuming ppo_{num_saved_timesteps}.zip run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    ppo.env.set_opponent(opponent)  # type: ignore

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_timesteps / TOTAL_TIMESTEPS
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo.learning_rate = calc_learning_rate
    # train
    callback = SaveAndReplaceOpponentCallback(102_400, num_saved_timesteps, ppo.n_steps)
    ppo = ppo.learn(
        TOTAL_TIMESTEPS - num_saved_timesteps, callback=callback, reset_num_timesteps=False
    )
    # evaluate
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    random = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    max_power = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    results = await agent.battle_against_multi([random, max_power, heuristic], n_battles=100)
    print(results)


if __name__ == "__main__":
    asyncio.run(train())
