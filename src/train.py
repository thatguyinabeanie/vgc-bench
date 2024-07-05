import os

from poke_env import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from callback import Callback
from env import ShowdownEnv, ShowdownVecEnvWrapper
from policy import MaskedActorCriticPolicy
from teams import TEAM1, TEAM2

BATTLE_FORMAT = "gen4ou"


def train(total_timesteps: int):
    # setup
    dummy_opponent = Agent(
        None,
        account_configuration=AccountConfiguration("DummyOpponent", None),
        battle_format=BATTLE_FORMAT,
        log_level=40,
        team=TEAM2,
    )
    env = ShowdownEnv(
        dummy_opponent,
        account_configuration=AccountConfiguration("Agent", None),
        battle_format=BATTLE_FORMAT,
        log_level=40,
        team=TEAM1,
    )
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    num_saved_timesteps = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_timesteps / total_timesteps
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo = PPO(
        MaskedActorCriticPolicy,
        wrapper_env,
        learning_rate=calc_learning_rate,
        tensorboard_log="output/logs/ppo",
    )
    if num_saved_timesteps > 0:
        ppo.set_parameters(os.path.join("output/saves", f"ppo_{num_saved_timesteps}.zip"))
        print(f"Resuming ppo_{num_saved_timesteps}.zip run.")
    # train
    callback = Callback(num_saved_timesteps, 102_400, BATTLE_FORMAT, TEAM1)
    ppo = ppo.learn(
        total_timesteps - num_saved_timesteps, callback=callback, reset_num_timesteps=False
    )


if __name__ == "__main__":
    train(100_000_000)
