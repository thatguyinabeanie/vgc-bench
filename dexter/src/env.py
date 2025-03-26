from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import numpy.typing as npt
import supersuit as ss
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStackObservation
from poke_env import ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import DoublesEnv, SimpleHeuristicsPlayer, SingleAgentWrapper
from src.agent import Agent
from src.teams import RandomTeamBuilder
from src.utils import (
    LearningStyle,
    battle_format,
    doubles_chunk_obs_len,
    frame_stack,
    moves,
    num_frames,
)
from stable_baselines3.common.monitor import Monitor


class ShowdownEnv(DoublesEnv[npt.NDArray[np.float32]]):
    _teampreview_draft: list[str]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metadata = {"name": "showdown_v1", "render_modes": ["human"]}
        self.render_mode: str | None = None
        self.observation_spaces = {
            agent: Box(-1, len(moves), shape=(12, doubles_chunk_obs_len), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._teampreview_draft = []

    @classmethod
    def create_env(
        cls, num_teams: int, port: int, device: str, learning_style: LearningStyle
    ) -> Env:
        env = cls(
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(list(range(num_teams)), battle_format),
            strict=False,
        )
        if learning_style == LearningStyle.PURE_SELF_PLAY:
            if frame_stack:
                env = ss.frame_stack_v2(env, stack_size=num_frames, stack_dim=0)
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(
                env, num_vec_envs=8, num_cpus=8, base_class="stable_baselines3"
            )
            return env  # type: ignore
        else:
            opponent = (
                Agent(
                    None,
                    num_frames=num_frames,
                    device=torch.device(device),
                    server_configuration=ServerConfiguration(
                        f"ws://localhost:{port}/showdown/websocket",
                        "https://play.pokemonshowdown.com/action.php?",
                    ),
                    battle_format=battle_format,
                    log_level=40,
                    accept_open_team_sheet=True,
                    open_timeout=None,
                    team=RandomTeamBuilder(list(range(num_teams)), battle_format),
                )
                if learning_style.is_self_play
                else SimpleHeuristicsPlayer(
                    server_configuration=ServerConfiguration(
                        f"ws://localhost:{port}/showdown/websocket",
                        "https://play.pokemonshowdown.com/action.php?",
                    ),
                    battle_format=battle_format,
                    log_level=40,
                    accept_open_team_sheet=True,
                    open_timeout=None,
                    team=RandomTeamBuilder(list(range(num_teams)), battle_format),
                )
            )
            env = SingleAgentWrapper(env, opponent)
            if frame_stack:
                env = FrameStackObservation(env, num_frames, padding_type="zero")
            env = Monitor(env)
            return env

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        result = super().reset(seed=seed, options=options)
        dead_tags = [k for k, b in self.agent1.battles.items() if b.finished]
        for tag in dead_tags:
            self.agent1._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                self.agent1.ps_client.send_message(f"/leave {tag}"), self.loop
            )
            self.agent2._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                self.agent2.ps_client.send_message(f"/leave {tag}"), self.loop
            )
        return result

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        elif battle.won:
            return 1
        elif battle.lost:
            return -1
        else:
            return 0

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        return Agent.embed_battle(battle, self._teampreview_draft, fake_ratings=True)
