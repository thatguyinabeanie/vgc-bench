import asyncio
import os
import random

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder


class Callback(BaseCallback):
    def __init__(self, num_envs: int, num_saved_timesteps: int, save_freq: int, battle_format: str):
        super().__init__()
        self.num_envs = num_envs
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
        self.policy_pool = [
            PPO.load(f"saves/{filename[:-4]}").policy for filename in os.listdir("saves")
        ]

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = random.choices(
            [MaskedActorCriticPolicy.clone(self.model)] + self.policy_pool, k=self.num_envs
        )
        opponents = self.model.env.env_method("get_opponent", indices=range(self.num_envs))
        for opponent, policy in zip(opponents, policies):
            opponent.policy = policy
        self.model.env.env_method("set_opponent", opponents, indices=range(self.num_envs))

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_freq == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.eval_agent.policy = new_policy
            results = asyncio.run(
                self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
            )
            self.model.logger.record("eval/random", results["RandomPlayer 1"][0])
            self.model.logger.record("eval/power", results["MaxBasePowerPlay 1"][0])
            self.model.logger.record("eval/heuristics", results["SimpleHeuristics 1"][0])
            self.policy_pool.append(new_policy)
            self.model.save(f"saves/ppo_{self.model.num_timesteps}")
