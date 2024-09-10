from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env import AccountConfiguration
from poke_env.environment import AbstractBattle, Battle, Move, Pokemon
from poke_env.player import (
    BattleOrder,
    Gen9EnvSinglePlayer,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


class ShowdownEnv(Gen9EnvSinglePlayer[npt.NDArray[np.float32], int]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @classmethod
    def create_env(cls, i: int, battle_format: str, self_play: bool) -> ShowdownEnv:
        if self_play:
            num_gpus = torch.cuda.device_count()
            opponent = Agent(
                None,
                device=torch.device(
                    f"cuda:{i % (num_gpus - 1) + 1}"
                    if num_gpus > 1
                    else "cuda" if num_gpus > 0 else "cpu"
                ),
                account_configuration=AccountConfiguration(f"Opponent{i + 1}", None),
                battle_format=battle_format,
                log_level=40,
                team=RandomTeamBuilder(battle_format),
            )
        else:
            opp_classes = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
            opponent = opp_classes[i % 3](
                account_configuration=AccountConfiguration(f"Opponent{i + 1}", None),
                battle_format=battle_format,
                log_level=40,
                team=RandomTeamBuilder(battle_format),
            )
        return cls(
            opponent,
            account_configuration=AccountConfiguration(f"Agent{i + 1}", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(battle_format),
        )

    def set_opp_policy(self, policy: MaskedActorCriticPolicy):
        assert isinstance(self._opponent, Agent)
        self._opponent.set_policy(policy)

    def get_opp_exp(self) -> tuple[npt.NDArray[np.float32], int, float, bool] | None:
        assert isinstance(self._opponent, Player)
        last_obs = self._opponent.last_obs
        assert isinstance(last_obs, Battle)
        action = self._opponent.action
        obs = self._opponent.current_obs
        assert last_obs is not None and obs is not None
        embedded_obs = Agent.embed_battle(last_obs)
        if action is None:
            action_space, *_ = np.nonzero(embedded_obs[:26] == 0)
            action_id = action_space[0].item() if len(action_space) > 0 else 0
        elif isinstance(action.order, Pokemon):
            action_id = [p.species for p in Agent.active_first(list(last_obs.team.values()))].index(
                action.order.species
            )
        elif isinstance(action.order, Move):
            if action.order.id == "struggle":
                action_id = 6
            else:
                assert last_obs.active_pokemon is not None
                action_id = list(last_obs.active_pokemon.moves.keys()).index(action.order.id) + 6
                if action.mega:
                    action_id += 4
                elif action.z_move:
                    action_id += 8
                elif action.dynamax:
                    action_id += 12
                elif action.terastallize:
                    action_id += 16
        else:
            action_id = -1
        reward = self.calc_reward(last_obs, obs)
        return embedded_obs, action_id, reward, last_obs.finished

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
        return Box(-1, 1, shape=(Agent.obs_len,), dtype=np.float32)
