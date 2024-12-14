import os
import pickle

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from imitation.util.logger import configure
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from src.policy import MaskedActorCriticPolicy
from src.utils import battle_format, device, env_class, num_frames, run_name
from stable_baselines3 import PPO


def pretrain():
    env = env_class(
        RandomPlayer(
            account_configuration=AccountConfiguration("DummyPlayer", None),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            start_listening=False,
        ),
        account_configuration=AccountConfiguration(f"DummyEnv", None),
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
    )
    ppo = PPO(MaskedActorCriticPolicy, env, policy_kwargs={"num_frames": num_frames}, device=device)
    with open("data/trajs.pkl", "rb") as f:
        trajs: list[Trajectory] = pickle.load(f)
    trajs = [
        Trajectory(
            obs=np.stack(
                [
                    np.stack(
                        [traj.obs[max(0, i + 1 + j - num_frames)] for j in range(num_frames)],  # type: ignore
                        axis=0,
                    )
                    for i in range(len(traj.obs))
                ],
                axis=0,
            ),
            acts=traj.acts,
            infos=None,
            terminal=True,
        )
        for traj in trajs
    ]
    bc = BC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(0),
        policy=ppo.policy,
        demonstrations=trajs,
        batch_size=1024,
        device=device,
        custom_logger=configure(f"logs/{run_name}", ["tensorboard"]),
    )
    bc.train(n_epochs=100)
    ppo.save(f"saves/{run_name}/0")


if __name__ == "__main__":
    if os.path.exists(f"saves/{run_name}/0.zip"):
        print("already have pretrained NN - aborting")
    else:
        pretrain()
