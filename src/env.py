from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env.environment import AbstractBattle
from poke_env.player import Gen4EnvSinglePlayer
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from agent import Agent


class ShowdownEnv(Gen4EnvSinglePlayer[npt.NDArray[np.float32], int]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def calc_reward(self, last_battle: AbstractBattle, current_battle: AbstractBattle) -> float:
        if not current_battle.finished:
            return 0
        elif current_battle.won:
            return 1
        elif current_battle.lost:
            return -1
        else:
            return 0

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        return np.array(Agent.embed_battle(battle))

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(0.0, 1.0, shape=(1414,), dtype=np.float32)


class ShowdownVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv: DummyVecEnv, env: ShowdownEnv):
        super().__init__(venv)
        self.env = env

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def set_opponent(self, opponent: Agent):
        self.env.set_opponent(opponent)
