import argparse
import os
import time
from subprocess import PIPE, STDOUT, Popen

from src.callback import Callback
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.utils import battle_format, behavior_clone, num_envs, num_frames, self_play, steps
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def train(num_teams: int, port: int, device: str):
    server = Popen(
        ["node", "pokemon-showdown", "start", str(port), "--no-security"],
        stdout=PIPE,
        stderr=STDOUT,
        cwd="pokemon-showdown",
    )
    time.sleep(10)
    env = SubprocVecEnv(
        [
            lambda i=i: ShowdownEnv.create_env(
                i, battle_format, num_frames, port, num_teams, self_play, device
            )
            for i in range(num_envs)
        ]
    )
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=2048 // num_envs,
        batch_size=64,
        gamma=1,
        ent_coef=0.02,
        tensorboard_log="logs",
        policy_kwargs={"num_frames": num_frames},
        device=device,
    )
    num_saved_timesteps = 0
    if (
        os.path.exists(f"saves/{num_teams}-teams")
        and len(os.listdir(f"saves/{num_teams}-teams")) > 0
    ):
        num_saved_timesteps = max(
            [int(file[:-4]) for file in os.listdir(f"saves/{num_teams}-teams")]
        )
        ppo.num_timesteps = num_saved_timesteps
        ppo.set_parameters(f"saves/{num_teams}-teams/{num_saved_timesteps}.zip", device=ppo.device)
    if behavior_clone:
        ppo.policy.actor_grad = num_saved_timesteps > 0  # type: ignore
    ppo.learn(
        steps,
        callback=Callback(num_teams, port),
        tb_log_name=f"{num_teams}-teams",
        reset_num_timesteps=False,
    )
    server.terminate()
    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Pokémon AI model")
    parser.add_argument("--num_teams", type=int, default=1, help="Number of teams to train with")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="CUDA device to use for training",
    )
    args = parser.parse_args()
    train(args.num_teams, args.port, args.device)
