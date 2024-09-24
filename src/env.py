from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import Space
from gymnasium.spaces import Box
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle
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
    def create_env(
        cls, i: int, battle_format: str, port: int, num_teams: int, self_play: bool
    ) -> ShowdownEnv:
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
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=40,
                team=RandomTeamBuilder(num_teams, battle_format),
            )
        else:
            opp_classes = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
            opponent = opp_classes[i % 3](
                account_configuration=AccountConfiguration(f"Opponent{i + 1}", None),
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=40,
                team=RandomTeamBuilder(num_teams, battle_format),
            )
        return cls(
            opponent,
            account_configuration=AccountConfiguration(f"Agent{i + 1}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(num_teams, battle_format),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        result = super().reset(seed=seed, options=options)
        assert isinstance(self._opponent, Player)
        tags = [k for k in self.battles.keys()]
        for tag in tags:
            if self.current_battle is None or tag != self.current_battle.battle_tag:
                self.agent._battles.pop(tag)
                self._opponent._battles.pop(tag)
        return result

    def set_opp_policy(self, policy: MaskedActorCriticPolicy):
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
        return Box(-1, 1, shape=(Agent.obs_len,), dtype=np.float32)
