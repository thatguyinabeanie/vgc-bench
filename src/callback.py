import asyncio
import json
import os
import random
import warnings

from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self, save_interval: int, battle_format: str):
        super().__init__()
        self.save_interval = save_interval
        self.policy_pool = [
            PPO.load(f"saves/{filename}").policy for filename in os.listdir("saves")
        ]
        with open("logs/progress.json") as f:
            json_objects = [json.loads(line) for line in f]
        self.win_rates = [
            json_obj["train/heuristics"]
            for json_obj in json_objects
            if "train/heuristics" in json_obj
        ]
        self.eval_agent = Agent(
            None,
            account_configuration=AccountConfiguration("EvalAgent", None),
            battle_format=battle_format,
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        )
        self.eval_opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("EvalOpponent", None),
            battle_format=battle_format,
            log_level=40,
            max_concurrent_battles=10,
            team=RandomTeamBuilder(),
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = random.choices(
            self.policy_pool + [MaskedActorCriticPolicy.clone(self.model)],
            weights=self.win_rates + [1],
            k=self.model.env.num_envs,
        )
        for i in range(self.model.env.num_envs):
            self.model.env.env_method("set_opp_policy", policies[i], indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_interval == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.eval_agent.set_policy(new_policy)
            asyncio.run(self.eval_agent.battle_against(self.eval_opponent, n_battles=100))
            win_rate = self.eval_agent.win_rate
            self.win_rates.append(win_rate)
            self.eval_agent.reset_battles()
            self.eval_opponent.reset_battles()
            self.model.logger.record("train/heuristics", win_rate)
            self.policy_pool.append(new_policy)
            self.model.save(f"saves/{self.model.num_timesteps}")
