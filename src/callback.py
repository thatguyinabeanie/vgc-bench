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
    def __init__(self, battle_format: str):
        super().__init__()
        self.eval_agent = Agent(
            None,
            account_configuration=AccountConfiguration("EvalAgent", None),
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
        self.policy_pool = [
            PPO.load(f"saves/{filename[:-4]}").policy for filename in os.listdir("saves")
        ]

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = [MaskedActorCriticPolicy.clone(self.model)] + self.policy_pool
        for i in range(self.model.env.num_envs):
            policy = random.choice(policies)
            self.model.env.env_method("set_opp_policy", policy, indices=i)

    def _on_training_end(self):
        new_policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.policy = new_policy
        results = asyncio.run(
            self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
        )
        self.model.logger.record("eval/random", results["Random"][0])
        self.model.logger.record("eval/power", results["Power"][0])
        self.model.logger.record("eval/heuristics", results["Heuristic"][0])
        self.model.logger.dump(self.model.num_timesteps)
        self.policy_pool.append(new_policy)
        self.model.save(f"saves/ppo_{self.model.num_timesteps}")
