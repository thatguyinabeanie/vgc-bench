from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import Space
from gymnasium.core import ActType
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import (
    BattleOrder,
    EnvPlayer,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


class ShowdownEnv(EnvPlayer[npt.NDArray[np.float32], ActType]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @classmethod
    def create_env(
        cls,
        i: int,
        battle_format: str,
        port: int,
        teams: list[int],
        opp_teams: list[int],
        self_play: bool,
        device: str,
    ) -> ShowdownEnv:
        if self_play:
            num_gpus = torch.cuda.device_count()
            other_gpu_ids = [f"cuda:{i}" for i in range(num_gpus) if f"cuda:{i}" != device]
            opponent = Agent(
                None,
                device=torch.device(
                    other_gpu_ids[i % len(other_gpu_ids)] if num_gpus > 0 else "cpu"
                ),
                account_configuration=AccountConfiguration(f"Opponent{port}-{i + 1}", None),
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=40,
                accept_open_team_sheet=True,
                team=RandomTeamBuilder(opp_teams, battle_format),
            )
        else:
            opp_classes = (
                [MaxBasePowerPlayer]
                if "vgc" in battle_format
                else [SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer]
            )
            opponent = opp_classes[i % len(opp_classes)](
                account_configuration=AccountConfiguration(f"Opponent{port}-{i + 1}", None),
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=40,
                accept_open_team_sheet=True,
                team=RandomTeamBuilder(opp_teams, battle_format),
            )
            opponent.teampreview = lambda _: "/team 123456"  # type: ignore
        return cls(
            opponent,
            account_configuration=AccountConfiguration(f"Agent{port}-{i + 1}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            team=RandomTeamBuilder(teams, battle_format),
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


class ShowdownSinglesEnv(ShowdownEnv[np.int64]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def action_to_move(self, action: np.int64, battle: AbstractBattle) -> BattleOrder | str:
        assert isinstance(battle, Battle)
        return Agent.singles_action_to_move_covering_teampreview(int(action), battle)

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(-1, 1, shape=(Agent.singles_obs_len,), dtype=np.float32)

    def describe_action(self) -> Space[np.int64]:
        return Discrete(Agent.singles_act_len)


class ShowdownDoublesEnv(ShowdownEnv[npt.NDArray[np.integer]]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def action_to_move(
        self, action: npt.NDArray[np.integer], battle: AbstractBattle
    ) -> BattleOrder | str:
        assert isinstance(battle, DoubleBattle)
        return Agent.doubles_action_to_move_covering_teampreview(action[0], action[1], battle)

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(-1, 1, shape=(Agent.doubles_obs_len,), dtype=np.float32)

    def describe_action(self) -> Space[npt.NDArray[np.integer]]:
        return MultiDiscrete([Agent.doubles_act_len, Agent.doubles_act_len])
