import asyncio
import json
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import numpy.typing as npt
from nashpy import Game
from poke_env import ServerConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.player import Player, SimpleHeuristicsPlayer
from src.agent import Agent
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import LearningStyle, battle_format, num_frames, steps
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(
        self,
        num_teams: int,
        port: int,
        device: str,
        learning_style: LearningStyle,
        behavior_clone: bool,
    ):
        super().__init__()
        self.learning_style = learning_style
        self.behavior_clone = behavior_clone
        self.num_teams = num_teams
        if not os.path.exists(
            f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}"
        ):
            os.mkdir(f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}")
        self.payoff_matrix: npt.NDArray[np.float32]
        self.prob_dist: list[float] | None = None
        if self.learning_style == LearningStyle.DOUBLE_ORACLE:
            if os.path.exists(
                f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams-payoff-matrix.json"
            ):
                with open(
                    f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams-payoff-matrix.json"
                ) as f:
                    self.payoff_matrix = np.array(json.load(f))
            else:
                self.payoff_matrix = np.array([[0]])
                with open(
                    f"results/logs{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams-payoff-matrix.json",
                    "w",
                ) as f:
                    json.dump(self.payoff_matrix.tolist(), f)
            g = Game(self.payoff_matrix)
            self.prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
        if self.learning_style.is_self_play:
            self.policy_pool = (
                []
                if not os.path.exists(
                    f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams"
                )
                else [
                    PPO.load(
                        f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams/{filename}",
                        device=device,
                    ).policy
                    for filename in os.listdir(
                        f"results/saves{'-bc' if behavior_clone else ''}{'-' + learning_style.abbrev}/{num_teams}-teams"
                    )
                ]
            )
        self.eval_agent = Agent(
            None,
            num_frames=num_frames,
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
        self.eval_agent2 = Agent(
            None,
            num_frames=num_frames,
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
        self.eval_opponent = SimpleHeuristicsPlayer(
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

    def _on_step(self) -> bool:
        assert self.model.env is not None
        dones: npt.NDArray[np.bool] = self.locals.get("dones")  # type: ignore
        for i, done in enumerate(dones):
            if done:
                policy = random.choices(self.policy_pool, weights=self.prob_dist, k=1)[0]
                self.model.env.env_method("set_opp_policy", policy, indices=i)
        return True

    def _on_training_start(self):
        assert self.model.env is not None
        if self.learning_style.is_self_play:
            if self.model.num_timesteps < steps:
                self.evaluate()
            if not self.policy_pool:
                self.model.save(
                    f"results/saves{'-bc' if self.behavior_clone else ''}{'-' + self.learning_style.abbrev}/{self.num_teams}-teams/{self.model.num_timesteps}"
                )
                policy = MaskedActorCriticPolicy.clone(self.model)
                self.policy_pool.append(policy)
            policies = random.choices(
                self.policy_pool, weights=self.prob_dist, k=self.model.env.num_envs
            )
            for i in range(self.model.env.num_envs):
                self.model.env.env_method("set_opp_policy", policies[i], indices=i)

    def _on_rollout_start(self):
        if self.behavior_clone:
            self.model.policy.actor_grad = self.model.num_timesteps >= steps  # type: ignore

    def _on_rollout_end(self):
        if self.model.num_timesteps % steps == 0:
            self.evaluate()
            if self.learning_style == LearningStyle.DOUBLE_ORACLE:
                self.update_payoff_matrix()
                g = Game(self.payoff_matrix)
                self.prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
            self.model.save(
                f"results/saves{'-bc' if self.behavior_clone else ''}{'-' + self.learning_style.abbrev}/{self.num_teams}-teams/{self.model.num_timesteps}"
            )
            if self.learning_style.is_self_play:
                policy = MaskedActorCriticPolicy.clone(self.model)
                self.policy_pool.append(policy)

    def evaluate(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        win_rate = self.compare(self.eval_agent, self.eval_opponent, 100)
        self.model.logger.record("train/eval", win_rate)

    def update_payoff_matrix(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        win_rates = np.array([])
        for p in self.policy_pool:
            self.eval_agent2.set_policy(deepcopy(p))
            win_rate = self.compare(self.eval_agent, self.eval_agent2, 100)
            win_rates = np.append(win_rates, round(2 * win_rate - 1, ndigits=2))
        self.payoff_matrix = np.concat([self.payoff_matrix, -win_rates.reshape(-1, 1)], axis=1)
        win_rates = np.append(win_rates, 0)
        self.payoff_matrix = np.concat([self.payoff_matrix, win_rates.reshape(1, -1)], axis=0)
        with open(
            f"results/logs{'-bc' if self.behavior_clone else ''}{'-' + self.learning_style.abbrev}/{self.num_teams}-teams-payoff-matrix.json",
            "w",
        ) as f:
            json.dump(self.payoff_matrix.tolist(), f)

    @staticmethod
    def compare(player1: Player, player2: Player, n_battles: int) -> float:
        asyncio.run(player1.battle_against(player2, n_battles=n_battles))
        win_rate = player1.win_rate
        dead_tags = [k for k, b in player1.battles.items() if b.finished]
        for tag in dead_tags:
            player1._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                player1.ps_client.send_message(f"/leave {tag}"), POKE_LOOP
            )
            player2._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                player2.ps_client.send_message(f"/leave {tag}"), POKE_LOOP
            )
        player1.reset_battles()
        player2.reset_battles()
        return win_rate
