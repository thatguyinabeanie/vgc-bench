import asyncio
import json
import os
import random

import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


class Callback(BaseCallback):
    def __init__(
        self, opponents: list[Agent], num_saved_timesteps: int, save_freq: int, battle_format: str
    ):
        super().__init__()
        self.opponents = opponents
        self.num_saved_timesteps = num_saved_timesteps
        self.save_freq = save_freq
        self.eval_agent = Agent(
            None,
            account_configuration=AccountConfiguration("EvalAgent", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )
        self.eval_opponents: list[Player] = [
            RandomPlayer(battle_format=battle_format, log_level=40, team=RandomTeamBuilder()),
            MaxBasePowerPlayer(battle_format=battle_format, log_level=40, team=RandomTeamBuilder()),
            SimpleHeuristicsPlayer(
                battle_format=battle_format, log_level=40, team=RandomTeamBuilder()
            ),
        ]
        self.elo_agent = Agent(
            None,
            account_configuration=AccountConfiguration("EloAgent", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )
        self.elo_opponent = Agent(
            None,
            account_configuration=AccountConfiguration("EloOpponent", None),
            battle_format=battle_format,
            log_level=40,
            team=RandomTeamBuilder(),
        )
        self.policy_pool = [
            PPO.load(f"saves/{filename[:-4]}").policy for filename in os.listdir("saves")
        ]
        self.elos: list[int] = []
        if os.path.exists("logs/elos.json"):
            with open("logs/elos.json") as f:
                self.elos = json.load(f)

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        assert len(self.policy_pool) == len(self.elos)
        for opponent in self.opponents:
            if not self.policy_pool:
                policy = MaskedActorCriticPolicy.clone(self.model)
            else:
                policy = random.choices(self.policy_pool, weights=self.elos, k=1)[0]
            opponent.policy = policy

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_freq == 0:
            # evaluate against baselines
            self.eval_agent.policy = MaskedActorCriticPolicy.clone(self.model)
            results = asyncio.run(
                self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
            )
            self.model.logger.record("eval/random", results["RandomPlayer 1"][0])
            self.model.logger.record("eval/power", results["MaxBasePowerPlay 1"][0])
            self.model.logger.record("eval/heuristics", results["SimpleHeuristics 1"][0])
            # save and append to saved policies
            self.model.save(f"saves/ppo_{self.model.num_timesteps}")
            self.policy_pool.append(PPO.load(f"saves/ppo_{self.model.num_timesteps}").policy)
            self.elos.append(1000)
            # update policies against each other, save and log
            self.update_elos(10 * len(self.policy_pool))
            with open("logs/elos.json", "w") as f:
                json.dump(self.elos, f)
            self.model.logger.record("eval/max-elo", max(self.elos))
            self.model.logger.record("eval/elo-variance", np.var(self.elos))

    def update_elos(self, n: int):
        if len(self.policy_pool) < 2:
            return
        for _ in range(n):
            [i, j, *_] = random.sample(range(len(self.policy_pool)), k=2)
            self.elo_agent.policy = self.policy_pool[i]
            self.elo_opponent.policy = self.policy_pool[j]
            win_rate, lose_rate = asyncio.run(self.elo_agent.battle_against(self.elo_opponent))
            score = 1 if win_rate == 1 else 0 if lose_rate == 1 else 0.5
            expectation = 1 / (1 + 10 ** ((self.elos[j] - self.elos[i]) / 400))
            delta = round(20 * (score - expectation))
            self.elos[i] = max(1, self.elos[i] + delta)
            self.elos[j] = max(1, self.elos[j] - delta)
