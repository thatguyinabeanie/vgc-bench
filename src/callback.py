import asyncio
import os
import random

import torch
from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


class Callback(BaseCallback):
    def __init__(
        self, num_saved_timesteps: int, battle_format: str, save_interval: int, bench_interval: int
    ):
        super().__init__()
        self.num_saved_timesteps = num_saved_timesteps
        self.save_interval = save_interval
        self.bench_interval = bench_interval
        self.eval_agent = Agent(
            None,
            account_configuration=AccountConfiguration("EvalAgent", None),
            battle_format=battle_format,
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        )
        self.eval_agent2 = Agent(
            None,
            account_configuration=AccountConfiguration("EvalAgent2", None),
            battle_format=battle_format,
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        )
        self.eval_opponents: list[Player] = [
            RandomPlayer(
                account_configuration=AccountConfiguration("Random", None),
                battle_format=battle_format,
                log_level=40,
                max_concurrent_battles=10,
                team=RandomTeamBuilder(),
            ),
            MaxBasePowerPlayer(
                account_configuration=AccountConfiguration("Power", None),
                battle_format=battle_format,
                log_level=40,
                max_concurrent_battles=10,
                team=RandomTeamBuilder(),
            ),
            SimpleHeuristicsPlayer(
                account_configuration=AccountConfiguration("Heuristic", None),
                battle_format=battle_format,
                log_level=40,
                max_concurrent_battles=10,
                team=RandomTeamBuilder(),
            ),
        ]
        self.policy_pool = [torch.load(f"saves/{filename}") for filename in os.listdir("saves")]

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.bench_interval == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.eval_agent.set_policy(new_policy)
            results = asyncio.run(
                self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
            )
            self.model.logger.record("eval/random", results["Random"][0])
            self.model.logger.record("eval/power", results["Power"][0])
            self.model.logger.record("eval/heuristics", results["Heuristic"][0])
        if self.model.num_timesteps % self.save_interval == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.eval_agent.set_policy(new_policy)
            if not self.policy_pool:
                self.policy_pool.append(new_policy)
                self.model.policy.save(f"saves/{self.model.num_timesteps}.pt")
            else:
                self.eval_agent2.set_policy(self.policy_pool[-1])
                win_rate, lose_rate = asyncio.run(
                    self.eval_agent.battle_against(self.eval_agent2, n_battles=100)
                )
                score = round(win_rate + (1 - win_rate - lose_rate) / 2, 5)
                self.model.logger.record("eval/score", score)
                if score >= 0.55:
                    self.policy_pool.append(new_policy)
                    self.model.policy.save(f"saves/{self.model.num_timesteps}.pt")
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = [MaskedActorCriticPolicy.clone(self.model)] + self.policy_pool
        for i in range(self.model.env.num_envs):
            policy = random.choice(policies)
            self.model.env.env_method("set_opp_policy", policy, indices=i)
