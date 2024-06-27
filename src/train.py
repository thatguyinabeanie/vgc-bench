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
    def __init__(self, rollout_len: int, save_freq: int, rollouts: int):
        super().__init__()
        self.rollout_len = rollout_len
        self.save_freq = save_freq
        self.rollout_count = rollouts

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        opponent = Agent(self.model.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        self.model.env.set_opponent(opponent)  # type: ignore
        self.rollout_count += 1
        if self.rollout_count % self.save_freq == 0:
            self.model.save(f"output/saves/ppo_{self.rollout_len * self.rollout_count}")


async def train():
    # setup
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    ppo = PPO("MlpPolicy", wrapper_env, tensorboard_log="output/logs/ppo")
    num_saved_rollouts = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_rollouts = max([int(file[4:-4]) for file in files])
        ppo.set_parameters(f"output/saves/ppo_{num_saved_rollouts}.zip")
        print(f"Resuming ppo_{num_saved_rollouts}.zip run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    ppo.env.set_opponent(opponent)  # type: ignore

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_rollouts / TOTAL_TIMESTEPS
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo.learning_rate = calc_learning_rate
    # train
    callback = SaveAndReplaceOpponentCallback(
        rollout_len=ppo.n_steps, save_freq=50, rollouts=num_saved_rollouts
    )
    ppo = ppo.learn(TOTAL_TIMESTEPS, callback=callback, reset_num_timesteps=False)
    # evaluate
    agent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    random = RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    max_power = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    results = await agent.battle_against_multi([random, max_power, heuristic], n_battles=100)
    print(results)


if __name__ == "__main__":
    asyncio.run(train())
