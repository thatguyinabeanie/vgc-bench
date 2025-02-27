import asyncio
import json
import os
import random
import warnings

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from src.agent import Agent
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import (
    battle_format,
    behavior_clone,
    num_frames,
    port,
    run_name,
    self_play,
    steps,
    teams,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        if not os.path.exists("logs"):
            os.mkdir("logs")
        self.win_rates: list[float]
        if os.path.exists(f"logs/{run_name}-win-rates.json"):
            with open(f"logs/{run_name}-win-rates.json") as f:
                self.win_rates = json.load(f)
        else:
            self.win_rates = []
            with open(f"logs/{run_name}-win-rates.json", "w") as f:
                json.dump(self.win_rates, f)
        if self_play:
            self.policy_pool = (
                []
                if not os.path.exists(f"saves/{run_name}")
                else [
                    PPO.load(f"saves/{run_name}/{filename}").policy
                    for filename in os.listdir(f"saves/{run_name}")
                ]
            )
        self.eval_agent = Agent(
            None,
            num_frames=num_frames,
            account_configuration=AccountConfiguration(f"EvalAgent{port}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format),
        )
        opp_class = MaxBasePowerPlayer if "vgc" in battle_format else SimpleHeuristicsPlayer
        self.eval_opponent = opp_class(
            account_configuration=AccountConfiguration(f"EvalOpponent{port}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format),
        )

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        if self_play and self.num_timesteps == 0:
            if behavior_clone:
                win_rate = self.evaluate()
                self.win_rates.append(win_rate)
                with open(f"logs/{run_name}-win-rates.json", "w") as f:
                    json.dump(self.win_rates, f)
                self.model.logger.record("train/eval", win_rate)
            else:
                assert self.model.env is not None
                for i in range(self.model.env.num_envs):
                    opp = RandomPlayer(
                        account_configuration=AccountConfiguration(f"RandomOpp{port}-{i}", None)
                    )
                    self.model.env.set_attr("opponent", opp, indices=i)

    def _on_rollout_start(self):
        if self_play:
            assert self.model.env is not None
            if self.policy_pool:
                policies = random.choices(self.policy_pool, k=self.model.env.num_envs)
                for i in range(self.model.env.num_envs):
                    self.model.env.env_method("set_opp_policy", policies[i], indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % steps == 0:
            win_rate = self.evaluate()
            self.win_rates.append(win_rate)
            with open(f"logs/{run_name}-win-rates.json", "w") as f:
                json.dump(self.win_rates, f)
            self.model.save(f"saves/{run_name}/{self.model.num_timesteps}")
            self.model.logger.record("train/eval", win_rate)
            if self_play:
                policy = MaskedActorCriticPolicy.clone(self.model)
                self.policy_pool.append(policy)

    def evaluate(self) -> float:
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        asyncio.run(self.eval_agent.battle_against(self.eval_opponent, n_battles=100))
        win_rate = self.eval_agent.win_rate
        self.eval_agent.reset_battles()
        self.eval_opponent.reset_battles()
        return win_rate
