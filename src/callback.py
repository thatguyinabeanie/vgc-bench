import asyncio
import time

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy


class Callback(BaseCallback):
    def __init__(self, num_saved_timesteps: int, save_freq: int, battle_format: str, team: str):
        super().__init__()
        self.num_saved_timesteps = num_saved_timesteps
        self.save_freq = save_freq
        self.opponent = Agent(
            None,
            account_configuration=AccountConfiguration("Opponent", None),
            battle_format=battle_format,
            log_level=40,
            team=team,
        )
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
        self.opponent.policy = MaskedActorCriticPolicy.clone(self.model)
        self.model.env.set_opponent(self.opponent)  # type: ignore

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_freq == 0:
            self.eval_agent.policy = MaskedActorCriticPolicy.clone(self.model)
            results = asyncio.run(
                self.eval_agent.battle_against_multi(self.eval_opponents, n_battles=100)
            )
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.model.num_timesteps}")
            print(f"Saved checkpoint ppo_{self.model.num_timesteps}.zip")
