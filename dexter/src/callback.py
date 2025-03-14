import asyncio
import json
import os
import random
import warnings

import numpy as np
import numpy.typing as npt
from poke_env import ServerConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from src.agent import Agent
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import battle_format, behavior_clone, num_frames, self_play, steps
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self, num_teams: int, port: int):
        super().__init__()
        self.num_teams = num_teams
        if not os.path.exists("logs"):
            os.mkdir("logs")
        self.win_rates: list[float]
        if os.path.exists(f"logs/{num_teams}-teams-win-rates.json"):
            with open(f"logs/{num_teams}-teams-win-rates.json") as f:
                self.win_rates = json.load(f)
        else:
            self.win_rates = []
            with open(f"logs/{num_teams}-teams-win-rates.json", "w") as f:
                json.dump(self.win_rates, f)
        if self_play:
            self.policy_pool = (
                []
                if not os.path.exists(f"saves/{num_teams}-teams")
                else [
                    PPO.load(f"saves/{num_teams}-teams/{filename}").policy
                    for filename in os.listdir(f"saves/{num_teams}-teams")
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
        opp_class = MaxBasePowerPlayer if "vgc" in battle_format else SimpleHeuristicsPlayer
        self.eval_opponent = opp_class(
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
                policy = random.choice(self.policy_pool)
                self.model.env.env_method("set_opp_policy", policy, indices=i)
        return True

    def _on_training_start(self):
        assert self.model.env is not None
        if self_play:
            if not self.policy_pool:
                self.on_rollout_end()
            elif behavior_clone and not self.win_rates:
                self.evaluate()
            policies = random.choices(self.policy_pool, k=self.model.env.num_envs)
            for i in range(self.model.env.num_envs):
                self.model.env.env_method("set_opp_policy", policies[i], indices=i)

    def _on_rollout_start(self):
        if behavior_clone:
            self.model.policy.actor_grad = self.model.num_timesteps > 0  # type: ignore

    def _on_rollout_end(self):
        if self.model.num_timesteps % steps == 0:
            self.evaluate()
            self.model.save(f"saves/{self.num_teams}-teams/{self.model.num_timesteps}")
            if self_play:
                policy = MaskedActorCriticPolicy.clone(self.model)
                self.policy_pool.append(policy)

    def evaluate(self):
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        asyncio.run(self.eval_agent.battle_against(self.eval_opponent, n_battles=100))
        win_rate = self.eval_agent.win_rate
        dead_tags = [k for k, b in self.eval_agent.battles.items() if b.finished]
        for tag in dead_tags:
            self.eval_agent._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                self.eval_agent.ps_client.send_message(f"/leave {tag}"), POKE_LOOP
            )
            self.eval_opponent._battles.pop(tag)
            asyncio.run_coroutine_threadsafe(
                self.eval_opponent.ps_client.send_message(f"/leave {tag}"), POKE_LOOP
            )
        self.eval_agent.reset_battles()
        self.eval_opponent.reset_battles()
        self.win_rates.append(win_rate)
        with open(f"logs/{self.num_teams}-teams-win-rates.json", "w") as f:
            json.dump(self.win_rates, f)
        self.model.logger.record("train/eval", win_rate)
