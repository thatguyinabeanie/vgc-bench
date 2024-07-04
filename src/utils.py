import asyncio
import time
from typing import Any

import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs

from agent import Agent


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.predict_values(obs)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        distribution = self.get_distribution(obs)
        values = self.predict_values(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        bool_mask = ~obs[:, :10].bool()  # type: ignore
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.action_net(latent_pi)
        mask = torch.zeros(bool_mask.size()).to(self.device)
        for i in range(mask.size(0)):
            mask[i] = (
                mask[i]
                if bool_mask[i].all()
                else mask[i].masked_fill(
                    bool_mask[i], -(torch.max(mean_actions[i]) - torch.min(mean_actions[i]) + 1)
                )
            )
        return self.action_dist.proba_distribution(action_logits=mean_actions + mask)


class Callback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        num_saved_timesteps: int,
        n_steps: int,
        battle_format: str,
        team: str,
        opp_team: str,
        self_play: bool = False,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.total_timesteps = num_saved_timesteps
        self.n_steps = n_steps
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
        return True

    def on_rollout_end(self):
        self.total_timesteps += self.n_steps
        if self.self_play:
            agent = Agent(
                self.model.policy, battle_format=self.battle_format, log_level=40, team=self.team
            )
            self.model.env.set_opponent(agent)  # type: ignore
        if self.total_timesteps % self.save_freq == 0:
            agent = Agent(
                self.model.policy, battle_format=self.battle_format, log_level=40, team=self.team
            )
            results = asyncio.run(agent.battle_against_multi(self.eval_opponents, n_battles=100))
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")
