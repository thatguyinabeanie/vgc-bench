import os
import random
import warnings

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import ShowdownEnv
from policy import MaskedActorCriticPolicy
from teams import RandomTeamBuilder

warnings.filterwarnings("ignore", category=UserWarning)


class Callback(BaseCallback):
    def __init__(self, num_saved_timesteps: int, save_interval: int, battle_format: str):
        super().__init__()
        self.num_saved_timesteps = num_saved_timesteps
        self.save_interval = save_interval
        self.policy_pool = [
            PPO.load(f"saves/{filename}").policy for filename in os.listdir("saves")
        ]
        eval_opponents: list[type[Player]] = [
            RandomPlayer,
            MaxBasePowerPlayer,
            SimpleHeuristicsPlayer,
        ]
        eval_env = SubprocVecEnv(
            [
                lambda i=i: ShowdownEnv.create_env(
                    i,
                    battle_format,
                    opponent=eval_opponents[i](
                        battle_format=battle_format, log_level=40, team=RandomTeamBuilder()
                    ),
                    username="EvalAgent",
                )
                for i in range(len(eval_opponents))
            ]
        )
        self.evaluator = PPO(
            MaskedActorCriticPolicy, eval_env, learning_rate=0, n_steps=1024, n_epochs=1
        )

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        self.model.num_timesteps = self.num_saved_timesteps

    def _on_rollout_start(self):
        assert self.model.env is not None
        policies = [MaskedActorCriticPolicy.clone(self.model)] + self.policy_pool
        for i in range(self.model.env.num_envs):
            policy = random.choice(policies)
            self.model.env.env_method("set_opp_policy", policy, indices=i)

    def _on_rollout_end(self):
        if self.model.num_timesteps % self.save_interval == 0:
            new_policy = MaskedActorCriticPolicy.clone(self.model)
            self.policy_pool.append(new_policy)
            self.model.save(f"saves/{self.model.num_timesteps}")
            self.evaluator.set_parameters(f"saves/{self.model.num_timesteps}")
            self.evaluator.learn(3072)
            assert self.evaluator.env is not None
            [random_winrate, power_winrate, heuristics_winrate, *_] = self.evaluator.env.get_attr(
                "win_rate"
            )
            self.model.logger.record("eval/random", round(random_winrate, 3))
            self.model.logger.record("eval/power", round(power_winrate, 3))
            self.model.logger.record("eval/heuristics", round(heuristics_winrate, 3))
            self.evaluator.env.env_method("reset_env")
