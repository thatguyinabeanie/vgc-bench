import os

from poke_env import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from agent import Agent
from callback import Callback
from env import ShowdownEnv, ShowdownVecEnvWrapper
from policy import MaskedActorCriticPolicy
from teams import TEAM1

BATTLE_FORMAT = "gen4ou"


def train(total_timesteps: int):
    # setup
    dummy_opponent = Agent(
        None,
        account_configuration=AccountConfiguration("DummyOpponent", None),
        battle_format=BATTLE_FORMAT,
        log_level=40,
        team=TEAM1,
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
        saved_prog_frac = num_saved_timesteps / total_timesteps
        return 10**-4.23 / (8 * (progress + saved_prog_frac) / (1 + saved_prog_frac) + 1) ** 1.5

    ppo = PPO(
        MaskedActorCriticPolicy,
        wrapper_env,
        learning_rate=calc_learning_rate,
        n_steps=8 * 512,
        batch_size=1024,
        n_epochs=7,
        gamma=0.9999,
        gae_lambda=0.754,
        clip_range=0.0829,
        clip_range_vf=0.0184,
        ent_coef=0.0588,
        vf_coef=0.4375,
        max_grad_norm=0.5430,
        tensorboard_log="output/logs/ppo",
        policy_kwargs={"activation_fn": nn.ReLU, "net_arch": MaskedActorCriticPolicy.arch},
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
    train(1_000_000)
