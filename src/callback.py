import asyncio
import time

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback

from agent import Agent
from policy import MaskedActorCriticPolicy


class Callback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        num_saved_timesteps: int,
        battle_format: str,
        team: str,
        opp_team: str,
        self_play: bool = False,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.total_timesteps = num_saved_timesteps
        self.battle_format = battle_format
        self.team = team
        self.self_play = self_play
        eval_team = team if self_play else opp_team
        self.eval_opponents: list[Player] = [
            RandomPlayer(battle_format=self.battle_format, log_level=40, team=eval_team),
            MaxBasePowerPlayer(battle_format=self.battle_format, log_level=40, team=eval_team),
            SimpleHeuristicsPlayer(battle_format=self.battle_format, log_level=40, team=eval_team),
        ]

    def _on_step(self) -> bool:
        self.total_timesteps += 1
        return True

    def _on_rollout_start(self):
        if self.self_play:
            agent = Agent(
                MaskedActorCriticPolicy.clone(self.model),
                battle_format=self.battle_format,
                log_level=40,
                team=self.team,
            )
            self.model.env.set_opponent(agent)  # type: ignore

    def _on_rollout_end(self):
        if self.total_timesteps % self.save_freq == 0:
            agent = Agent(
                MaskedActorCriticPolicy.clone(self.model),
                battle_format=self.battle_format,
                log_level=40,
                team=self.team,
            )
            results = asyncio.run(agent.battle_against_multi(self.eval_opponents, n_battles=100))
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")
