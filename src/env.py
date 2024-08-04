from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder, Gen9EnvSinglePlayer
from stable_baselines3.common.policies import ActorCriticPolicy

from agent import Agent


class ShowdownEnv(Gen9EnvSinglePlayer[npt.NDArray[np.float32], int]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def set_opp_policy(self, policy: ActorCriticPolicy):
        assert isinstance(self._opponent, Agent)
        self._opponent.policy = policy.to(self._opponent.policy.device)

    @staticmethod
    def action_to_move(action: int, battle: AbstractBattle) -> BattleOrder:
        return Agent.action_to_move(action, battle)

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
        return Agent.embed_battle(battle)

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(0.0, 1.0, shape=(Agent.obs_len,), dtype=np.float32)
