import asyncio
import time
from typing import Any

import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from agent import Agent


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        action_space = [i for i, o in enumerate(obs[0][:10].tolist()) if o == 1]  # type: ignore
        mask = torch.full((10,), float("-inf")).to(self.device)
        mask[action_space] = 0
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_space:
            probs = torch.softmax(distribution.distribution.probs[0] + mask, dim=0)  # type: ignore
        else:
            probs = distribution.distribution.probs[0]  # type: ignore
        actions = torch.multinomial(probs, 1)  # type: ignore
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


class Callback(BaseCallback):
    def __init__(
        self, save_freq: int, num_saved_timesteps: int, n_steps: int, battle_format: str, team: str
    ):
        super().__init__()
        self.save_freq = save_freq
        self.total_timesteps = num_saved_timesteps
        self.n_steps = n_steps
        self.battle_format = battle_format
        self.team = team
        self.eval_opponents: list[Player] = [
            RandomPlayer(battle_format=self.battle_format, team=self.team),
            MaxBasePowerPlayer(battle_format=self.battle_format, team=self.team),
            SimpleHeuristicsPlayer(battle_format=self.battle_format, team=self.team),
        ]

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        agent = Agent(self.model.policy, battle_format=self.battle_format, team=self.team)
        self.model.env.set_opponent(agent)  # type: ignore
        self.total_timesteps += self.n_steps
        if self.total_timesteps % self.save_freq == 0:
            results = asyncio.run(agent.battle_against_multi(self.eval_opponents, n_battles=100))
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")
