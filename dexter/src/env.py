from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import Space
from gymnasium.core import ActType
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.wrappers.frame_stack import FrameStack
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
from src.agent import Agent
from src.constants import (
    doubles_act_len,
    doubles_chunk_obs_len,
    singles_act_len,
    singles_chunk_obs_len,
)
from src.data import moves
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder


class ShowdownEnv(EnvPlayer[npt.NDArray[np.float32], ActType]):
    _teampreview_draft: list[str]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._teampreview_draft = []

    @classmethod
    def create_env(
        cls,
        i: int,
        battle_format: str,
        num_frames: int,
        port: int,
        teams: list[int],
        opp_teams: list[int],
        self_play: bool,
        device: str,
    ) -> FrameStack:
        if self_play:
            num_gpus = torch.cuda.device_count()
            other_gpu_ids = [f"cuda:{i}" for i in range(num_gpus) if f"cuda:{i}" != device]
            opponent = Agent(
                None,
                num_frames=num_frames,
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
        env = cls(
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
        return FrameStack(env, num_frames)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        assert isinstance(self._opponent, Player)
        if isinstance(self._opponent, Agent):
            self._opponent.frames.clear()
        result = super().reset(seed=seed, options=options)
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
        return Agent.embed_battle(battle, self._teampreview_draft)


class ShowdownSinglesEnv(ShowdownEnv[np.int64]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def action_to_move(self, action: np.int64, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle)
        return Agent.singles_action_to_move(int(action), battle)

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(-1, len(moves), shape=(12, singles_chunk_obs_len), dtype=np.float32)

    def describe_action(self) -> Space[np.int64]:
        return Discrete(singles_act_len)


class ShowdownDoublesEnv(ShowdownEnv[npt.NDArray[np.integer]]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def action_to_move(
        self, action: npt.NDArray[np.integer], battle: AbstractBattle
    ) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        if isinstance(action, int):
            assert action == -1
            action = np.array([-1, -1])
        elif battle.teampreview:
            if len(self._teampreview_draft) == 4:
                self._teampreview_draft = []
            team_names = [p.name for p in battle.team.values()]
            self._teampreview_draft += [team_names[action[0] - 1], team_names[action[1] - 1]]
        return Agent.doubles_action_to_move(action[0], action[1], battle)

    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        return Box(-1, len(moves), shape=(12, doubles_chunk_obs_len), dtype=np.float32)

    def describe_action(self) -> Space[npt.NDArray[np.integer]]:
        return MultiDiscrete([doubles_act_len, doubles_act_len])
