import os

from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from env import ShowdownEnv, ShowdownVecEnvWrapper
from utils import Callback, MaskedActorCriticPolicy

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


def train(total_timesteps: int, self_play: bool):
    # setup
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM2)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM1)
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    ppo = PPO(MaskedActorCriticPolicy, wrapper_env, tensorboard_log="output/logs/ppo")
    num_saved_timesteps = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])
        ppo.set_parameters(os.path.join("output/saves", f"ppo_{num_saved_timesteps}.zip"))
        print(f"Resuming ppo_{num_saved_timesteps}.zip run.")
    if self_play:
        opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM1)
        ppo.env.set_opponent(opponent)  # type: ignore

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_timesteps / total_timesteps
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo.learning_rate = calc_learning_rate
    # train
    callback = Callback(
        102_400, num_saved_timesteps, ppo.n_steps, BATTLE_FORMAT, TEAM1, TEAM2, self_play
    )
    ppo = ppo.learn(
        total_timesteps - num_saved_timesteps, callback=callback, reset_num_timesteps=False
    )


if __name__ == "__main__":
    train(100_000_000, True)
