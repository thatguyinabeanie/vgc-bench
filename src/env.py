from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env import AccountConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder, Gen9EnvSinglePlayer
from stable_baselines3.common.policies import ActorCriticPolicy

from agent import Agent
from teams import RandomTeamBuilder


class ShowdownEnv(Gen9EnvSinglePlayer[npt.NDArray[np.float32], int]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @classmethod
    def create_env(cls, i: int, battle_format: str):
        num_gpus = torch.cuda.device_count()
        opponent = Agent(
            None,
            device=torch.device(f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu"),
            account_configuration=AccountConfiguration(f"Opponent{i + 1}", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )
        return cls(
            opponent,
            account_configuration=AccountConfiguration(f"Agent{i + 1}", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )

    def set_opp_policy(self, policy: ActorCriticPolicy):
        assert isinstance(self._opponent, Agent)
        self._opponent.set_policy(policy)

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
