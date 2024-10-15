import asyncio
import json
import os
import random
import warnings

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(
        self,
        save_interval: int,
        battle_format: str,
        teams: list[int],
        opp_teams: list[int],
        port: int,
        self_play: bool,
    ):
        super().__init__()
        self.save_interval = save_interval
        self.self_play = self_play
        self.run_name = (
            f"{','.join([str(t) for t in teams])}|{','.join([str(t) for t in opp_teams])}"
        )
        if self_play:
            self.policy_pool = (
                []
                if not os.path.exists(f"saves/{self.run_name}")
                else [
                    PPO.load(f"saves/{self.run_name}/{filename}").policy
                    for filename in os.listdir(f"saves/{self.run_name}")
                ]
            )
            with open(f"logs/{self.run_name}-win_rates.json") as f:
                self.win_rates = json.load(f)
        self.eval_agent = Agent(
            None,
            account_configuration=AccountConfiguration(
                f"EvalAgent{','.join([str(t) for t in teams])}", None
            ),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            team=RandomTeamBuilder(teams, battle_format),
        )
        opp_class = MaxBasePowerPlayer if "vgc" in battle_format else SimpleHeuristicsPlayer
        self.eval_opponent = opp_class(
            account_configuration=AccountConfiguration(
                f"EvalOpponent{','.join([str(t) for t in opp_teams])}", None
            ),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            team=RandomTeamBuilder(opp_teams, battle_format),
        )
        self.eval_opponent.teampreview = Agent.teampreview_

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        if self.self_play:
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
            self.eval_agent.reset_battles()
            self.eval_opponent.reset_battles()
            self.model.save(f"saves/{self.run_name}/{self.model.num_timesteps}")
            self.model.logger.record("train/eval", win_rate)
            if self.self_play:
                self.policy_pool.append(new_policy)
                self.win_rates.append(win_rate)
                with open(f"logs/{self.run_name}-win_rates.json", "w") as f:
                    json.dump(self.win_rates, f)
