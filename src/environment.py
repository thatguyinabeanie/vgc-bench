from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import Gen4EnvSinglePlayer
from poke_env.ps_client import AccountConfiguration

from agent import Agent


class ShowdownEnv(Gen4EnvSinglePlayer[npt.NDArray[np.float32], int]):
    def __init__(self, opponent: Agent, agent_config: AccountConfiguration | None = None):
        super().__init__(
            opponent=opponent,
            account_configuration=agent_config,
            battle_format="gen4randombattle",
            log_level=40,
        )

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
        if isinstance(battle, Battle):
            return np.array(Agent.embed_battle(battle))
        elif isinstance(battle, DoubleBattle):
            return np.array([])
        else:
            raise TypeError()

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(0.0, 1.0, shape=(1404,))
