from __future__ import annotations

from typing import Any

import torch
from gymnasium.spaces import Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[256, 256, 256, 256, 256],
            features_extractor_class=ReduceTextEmbeddingDim,
        )

    @classmethod
    def clone(cls, model: BaseAlgorithm) -> MaskedActorCriticPolicy:
        new_policy = cls(model.observation_space, model.action_space, model.lr_schedule)
        new_policy.load_state_dict(model.policy.state_dict())
        return new_policy

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.get_distribution_and_values(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        distribution, values = self.get_distribution_and_values(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution_and_values(self, obs: PyTorchObs) -> tuple[Distribution, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, latent_vf = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.action_net(latent_pi)
        mask = obs[:, :26]  # type: ignore
        mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)
        mask = torch.where(mask == 1, float("-inf"), mask)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions + mask)
        values = self.value_net(latent_vf)
        return distribution, values


class ReduceTextEmbeddingDim(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space[Any]):
        super().__init__(observation_space, features_dim=3304)
        self.n_input = 300
        self.linear = torch.nn.Linear(self.n_input, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x[:, :556]
        for i in range(12):
            a = 556 + i * (65 + 2 * self.n_input)
            mon_output = torch.cat(
                [
                    x[:, a : a + 61],
                    self.linear(x[:, a + 61 : a + 61 + self.n_input]),
                    self.linear(x[:, a + 61 + self.n_input : a + 61 + 2 * self.n_input]),
                ],
                dim=1,
            )
            for j in range(4):
                b = a + 61 + 2 * self.n_input + j * (23 + self.n_input)
                move_output = torch.cat(
                    [x[:, b : b + 24], self.linear(x[:, b + 24 : b + 24 + self.n_input])], dim=1
                )
                mon_output = torch.cat([mon_output, move_output], dim=1)
            output = torch.cat([output, mon_output], dim=1)
        return output
