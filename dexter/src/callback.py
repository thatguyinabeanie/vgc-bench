import asyncio
import json
import os
import random
import warnings

import numpy as np
import numpy.typing as npt
import torch
from nashpy import Game
from poke_env.player import MaxBasePowerPlayer, Player
from poke_env.ps_client import AccountConfiguration, ServerConfiguration
from src.agent import Agent
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder, TeamToggle
from src.utils import LearningStyle, allow_mirror_match, battle_format, steps
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(
        self,
        teams: list[int],
        port: int,
        device: str,
        learning_style: LearningStyle,
        behavior_clone: bool,
        num_frames: int,
    ):
        super().__init__()
        self.learning_style = learning_style
        self.behavior_clone = behavior_clone
        self.teams = teams
        self.run_ident = "".join(
            [
                "-bc" if behavior_clone else "",
                f"-fs{num_frames}" if num_frames > 1 else "",
                "-" + learning_style.abbrev,
                "-xm" if not allow_mirror_match else "",
            ]
        )[1:]
        if not os.path.exists(f"results/logs-{self.run_ident}"):
            os.mkdir(f"results/logs-{self.run_ident}")
        toggle = None if allow_mirror_match else TeamToggle(len(teams))
        self.eval_agent = Agent(
            num_frames,
            torch.device(device),
            account_configuration=AccountConfiguration.randgen(10),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            max_concurrent_battles=10,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )
        self.eval_agent2 = Agent(
            num_frames,
            torch.device(device),
            account_configuration=AccountConfiguration.randgen(10),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            max_concurrent_battles=10,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )
        self.eval_opponent = MaxBasePowerPlayer(
            account_configuration=AccountConfiguration.randgen(10),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            max_concurrent_battles=10,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.payoff_matrix: npt.NDArray[np.float32]
        self.prob_dist: list[float] | None = None
        self.best_indices: list[int] | None = None
        if self.learning_style == LearningStyle.DOUBLE_ORACLE:
            if os.path.exists(
                f"results/logs-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams-payoff-matrix.json"
            ):
                with open(
                    f"results/logs-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams-payoff-matrix.json"
                ) as f:
                    payoff_matrix, self.best_indices = json.load(f)
                    self.payoff_matrix = np.array(payoff_matrix)
            else:
                self.payoff_matrix = np.array([[0]])
                self.best_indices = [0]
            g = Game(self.payoff_matrix)
            self.prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
        if self.learning_style.is_self_play:
            if self.model.num_timesteps < steps:
                self.evaluate()
            if not (
                os.path.exists(
                    f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
                )
                and os.listdir(
                    f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
                )
            ):
                assert not self.behavior_clone, "make sure to provide an initial BC agent"
                self.model.save(
                    f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{self.model.num_timesteps}"
                )

    def _on_rollout_start(self):
        self.model.logger.dump(self.model.num_timesteps)
        if self.behavior_clone:
            self.model.policy.actor_grad = self.model.num_timesteps >= steps  # type: ignore
        if (
            self.learning_style == LearningStyle.FICTITIOUS_PLAY
            or self.learning_style == LearningStyle.DOUBLE_ORACLE
        ):
            assert self.model.env is not None
            policy_files = os.listdir(
                f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
            )
            if self.learning_style == LearningStyle.DOUBLE_ORACLE:
                assert self.best_indices is not None
                policy_files = [policy_files[i] for i in self.best_indices]
            policies = random.choices(
                policy_files, weights=self.prob_dist, k=self.model.env.num_envs
            )
            for i in range(self.model.env.num_envs):
                self.model.env.env_method("cleanup", indices=i)
                policy = PPO.load(
                    f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{policies[i]}",
                    device=self.model.device,
                ).policy
                self.model.env.env_method("set_opp_policy", policy, indices=i)

    def _on_training_end(self):
        self.evaluate()
        self.model.logger.dump(self.model.num_timesteps)
        self.model.save(
            f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{self.model.num_timesteps}"
        )
        if self.learning_style == LearningStyle.DOUBLE_ORACLE:
            self.update_payoff_matrix()

    def evaluate(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        win_rate = self.compare(self.eval_agent, self.eval_opponent, 100)
        self.model.logger.record("train/eval", win_rate)

    def update_payoff_matrix(self):
        assert self.best_indices is not None
        policy_files = os.listdir(
            f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
        )
        policy = PPO.load(
            f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{policy_files[-1]}",
                device=self.model.device,
        ).policy
        self.eval_agent.set_policy(policy)
        win_rates = np.array([])
        best_policy_files = [policy_files[i] for i in self.best_indices]
        for p in best_policy_files:
            policy2 = PPO.load(
                f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{p}",
                device=self.model.device,
            ).policy
            self.eval_agent2.set_policy(policy2)
            win_rate = self.compare(self.eval_agent, self.eval_agent2, 100)
            win_rates = np.append(win_rates, round(2 * win_rate - 1, ndigits=2))
        self.payoff_matrix = np.concat([self.payoff_matrix, -win_rates.reshape(-1, 1)], axis=1)
        win_rates = np.append(win_rates, 0)
        self.payoff_matrix = np.concat([self.payoff_matrix, win_rates.reshape(1, -1)], axis=0)
        self.best_indices += [len(policy_files) - 1]
        if len(self.best_indices) > 8:
            g = Game(self.payoff_matrix)
            prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
            worst_index = np.argmin(prob_dist)
            self.best_indices.pop(worst_index)
            self.payoff_matrix = np.delete(self.payoff_matrix, worst_index, axis=0)
            self.payoff_matrix = np.delete(self.payoff_matrix, worst_index, axis=1)
            assert len(self.best_indices) == 8
            assert self.payoff_matrix.shape == (8, 8)
        with open(
            f"results/logs-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams-payoff-matrix.json",
            "w",
        ) as f:
            json.dump((self.payoff_matrix.tolist(), self.best_indices), f)

    @staticmethod
    def compare(player1: Player, player2: Player, n_battles: int) -> float:
        asyncio.run(player1.battle_against(player2, n_battles=n_battles))
        win_rate = player1.win_rate
        player1.reset_battles()
        player2.reset_battles()
        return win_rate
