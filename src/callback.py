import asyncio
import os
import random

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy


class Callback(BaseCallback):
    def __init__(
        self,
        opponents: list[Agent],
        num_saved_timesteps: int,
        save_freq: int,
        battle_format: str,
        team: str,
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
            team=team,
        )
        self.eval_opponents: list[Player] = [
            RandomPlayer(battle_format=battle_format, log_level=40, team=team),
            MaxBasePowerPlayer(battle_format=battle_format, log_level=40, team=team),
            SimpleHeuristicsPlayer(battle_format=battle_format, log_level=40, team=team),
        ]

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        saves = os.listdir("output/saves") if os.path.exists("output/saves") else []
        weights = [i + 1 for i in range(len(saves))]
        for opponent in self.opponents[1:]:
            if random.random() < (len(saves) + 1) / (sum(weights) + len(saves) + 1):
                model = self.model
            else:
                model = PPO(MaskedActorCriticPolicy, self.model.env)
                opponent_file = random.choices(saves, weights=weights, k=1)[0]
                model.set_parameters(os.path.join("output/saves", opponent_file))
            opponent.policy = MaskedActorCriticPolicy.clone(model)

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_freq == 0:
            self.eval_agent.policy = MaskedActorCriticPolicy.clone(self.model)
            results = asyncio.run(
                self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
            )
            self.model.logger.record("eval/random", results["RandomPlayer 1"][0])
            self.model.logger.record("eval/power", results["MaxBasePowerPlay 1"][0])
            self.model.logger.record("eval/heuristics", results["SimpleHeuristics 1"][0])
            self.model.save(f"output/saves/ppo_{self.model.num_timesteps}")
            print(f"Saved checkpoint ppo_{self.model.num_timesteps}.zip")
