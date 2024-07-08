import os

from poke_env import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from agent import Agent
from callback import Callback
from env import ShowdownEnv
from policy import MaskedActorCriticPolicy
from teams import TEAM1


def train():
    # setup
    total_steps = 10_000_000
    steps = 102_400
    num_envs = 8
    battle_format = "gen4ou"
    opponents = [
        Agent(
            None,
            account_configuration=AccountConfiguration(f"Opponent{i}", None),
            battle_format=battle_format,
            log_level=40,
            team=TEAM1,
        )
        for i in range(num_envs)
    ]
    env = DummyVecEnv(
        [
            lambda i=i: ShowdownEnv(
                opponents[i],
                account_configuration=AccountConfiguration(f"Agent{i}", None),
                battle_format=battle_format,
                log_level=40,
                team=TEAM1,
            )
            for i in range(num_envs)
        ]
    )
    num_saved_timesteps = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = (1 - progress_remaining) * (num_saved_timesteps + steps) / total_steps
        return 10**-4.23 / (8 * progress + 1) ** 1.5

    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=calc_learning_rate,
        n_steps=4096 // num_envs,
        batch_size=256,
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
    # train
    callback = Callback(opponents, num_saved_timesteps, steps, battle_format, TEAM1)
    ppo = ppo.learn(num_saved_timesteps + steps, callback=callback, reset_num_timesteps=False)


if __name__ == "__main__":
    train()
