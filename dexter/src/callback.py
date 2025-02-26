import asyncio
import json
import os
import random
import warnings

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from src.agent import Agent
from src.policy import MaskedActorCriticPolicy
from src.teams import RandomTeamBuilder
from src.utils import battle_format, num_frames, port, self_play, steps, teams
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.run_name = ",".join([str(t) for t in teams])
        if not os.path.exists("logs"):
            os.mkdir("logs")
        if os.path.exists(f"logs/{self.run_name}-win-rates.json"):
            with open(f"logs/{self.run_name}-win-rates.json") as f:
                self.win_rates = json.load(f)
        else:
            self.win_rates = []
            with open(f"logs/{self.run_name}-win-rates.json", "w") as f:
                json.dump(self.win_rates, f)
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

    # def _on_training_start(self):
    #     if self.self_play and len(self.policy_pool) == 1 and len(self.win_rates) == 0:
    #         win_rate = self.evaluate()
    #         self.win_rates.append(win_rate)
    #         with open(f"logs/{self.run_name}-win-rates.json", "w") as f:
    #             json.dump(self.win_rates, f)
    #         self.model.logger.record("train/eval", win_rate)

    def _on_rollout_start(self):
        if self_play:
            assert self.model.env is not None
            num_policies = (
                len(os.listdir(f"saves/{self.run_name}"))
                if os.path.exists(f"saves/{self.run_name}")
                else 0
            )
            if num_policies == 0:
                policies = [
                    MaskedActorCriticPolicy.clone(self.model)
                    for _ in range(self.model.env.num_envs)
                ]
            else:
                filenames = random.choices(
                    os.listdir(f"saves/{self.run_name}"), k=self.model.env.num_envs
                )
                policies = [
                    PPO.load(f"saves/{self.run_name}/{filename}").policy for filename in filenames
                ]
            for i in range(self.model.env.num_envs):
                self.model.env.env_method("set_opp_policy", policies[i], indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % steps == 0:
            win_rate = self.evaluate()
            self.win_rates.append(win_rate)
            with open(f"logs/{self.run_name}-win-rates.json", "w") as f:
                json.dump(self.win_rates, f)
            self.model.save(f"saves/{self.run_name}/{self.model.num_timesteps}")
            self.model.logger.record("train/eval", win_rate)

    def evaluate(self) -> float:
        policy = MaskedActorCriticPolicy.clone(self.model)
        self.eval_agent.set_policy(policy)
        asyncio.run(self.eval_agent.battle_against(self.eval_opponent, n_battles=100))
        win_rate = self.eval_agent.win_rate
        self.eval_agent.reset_battles()
        self.eval_opponent.reset_battles()
        return win_rate
