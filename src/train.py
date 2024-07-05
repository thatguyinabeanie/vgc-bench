import os
from copy import deepcopy

from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from callback import Callback
from env import ShowdownEnv, ShowdownVecEnvWrapper
from policy import MaskedActorCriticPolicy
from teams import TEAM1, TEAM2

BATTLE_FORMAT = "gen4ou"


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
        opponent = Agent(
            deepcopy(ppo.policy), battle_format=BATTLE_FORMAT, log_level=40, team=TEAM1
        )
        ppo.env.set_opponent(opponent)  # type: ignore

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_timesteps / total_timesteps
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo.learning_rate = calc_learning_rate
    # train
    callback = Callback(102_400, num_saved_timesteps, BATTLE_FORMAT, TEAM1, TEAM2, self_play)
    ppo = ppo.learn(
        total_timesteps - num_saved_timesteps, callback=callback, reset_num_timesteps=False
    )


if __name__ == "__main__":
    train(100_000_000, True)
