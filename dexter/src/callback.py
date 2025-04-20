import asyncio
import json
import os
import random
import warnings

import torch
import numpy as np
import numpy.typing as npt
from nashpy import Game
from poke_env import ServerConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.player import MaxBasePowerPlayer, Player
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
        self.payoff_matrix: npt.NDArray[np.float32]
        self.prob_dist: list[float] | None = None
        if self.learning_style == LearningStyle.DOUBLE_ORACLE:
            if os.path.exists(
                f"results/logs-{self.run_ident}/{','.join([str(t) for t in teams])}-teams-payoff-matrix.json"
            ):
                with open(
                    f"results/logs-{self.run_ident}/{','.join([str(t) for t in teams])}-teams-payoff-matrix.json"
                ) as f:
                    self.payoff_matrix = np.array(json.load(f))
            else:
                self.payoff_matrix = np.array([[0]])
                with open(
                    f"results/logs-{self.run_ident}/{','.join([str(t) for t in teams])}-teams-payoff-matrix.json",
                    "w",
                ) as f:
                    json.dump(self.payoff_matrix.tolist(), f)
            g = Game(self.payoff_matrix)
            self.prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
        toggle = None if allow_mirror_match else TeamToggle(len(teams))
        self.eval_agent = Agent(
            num_frames,
            torch.device(device),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )
        self.eval_agent2 = Agent(
            num_frames,
            torch.device(device),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )
        self.eval_opponent = MaxBasePowerPlayer(
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format, toggle),
        )

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
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
        if self.learning_style.is_self_play:
            assert self.model.env is not None
            policy_files = os.listdir(
                f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
            )
            policies = random.choices(policy_files, weights=self.prob_dist, k=self.model.env.num_envs)
            for i in range(self.model.env.num_envs):
                win_rate: float = self.model.env.env_method("get_win_rate", indices=i)[0]
                self.model.env.env_method("cleanup", indices=i)
                if win_rate == 0 or win_rate > 0.5:
                    policy = PPO.load(
                        f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{policies[i]}",
                        device=self.model.device,
                    ).policy
                    self.model.env.env_method("set_opp_policy", policy, indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % steps == 0:
            self.evaluate()
            if self.learning_style == LearningStyle.DOUBLE_ORACLE:
                self.update_payoff_matrix()
                g = Game(self.payoff_matrix)
                self.prob_dist = list(g.support_enumeration())[0][0].tolist()  # type: ignore
            self.model.save(
                f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams/{self.model.num_timesteps}"
            )

    def evaluate(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        win_rate = self.compare(self.eval_agent, self.eval_opponent, 100)
        self.model.logger.record("train/eval", win_rate)

    def update_payoff_matrix(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        win_rates = np.array([])
        for p in os.listdir(
            f"results/saves-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams"
        ):
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
        with open(
            f"results/logs-{self.run_ident}/{','.join([str(t) for t in self.teams])}-teams-payoff-matrix.json",
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
