from __future__ import annotations

from typing import Any

import torch
from gymnasium import Space
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
            features_extractor_class=EmbeddingFeaturesExtractor,
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
        features = self.extract_features(obs)  # type: ignore
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, latent_vf = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.action_net(latent_pi)
        mask = obs[:, :26]  # type: ignore
        mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)  # type: ignore
        mask = torch.where(mask == 1, float("-inf"), mask)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions + mask)
        values = self.value_net(latent_vf)
        return distribution, values


class EmbeddingFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space[Any]):
        super().__init__(observation_space, features_dim=4984)
        self.ability_desc_layer = torch.nn.Linear(768, 32)
        self.item_desc_layer = torch.nn.Linear(768, 32)
        self.move_desc_layer = torch.nn.Linear(768, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x[:, :556]
        for i in range(12):
            a = 556 + 4785 * i
            mon_output = torch.cat(
                [
                    self.ability_desc_layer(x[:, a : a + 768]),
                    self.item_desc_layer(x[:, a + 768 : a + 2 * 768]),
                    self.move_desc_layer(x[:, a + 2 * 768 : a + 3 * 768]),
                    self.move_desc_layer(x[:, a + 3 * 768 : a + 4 * 768]),
                    self.move_desc_layer(x[:, a + 4 * 768 : a + 5 * 768]),
                    self.move_desc_layer(x[:, a + 5 * 768 : a + 6 * 768]),
                    x[:, a + 6 * 768 : a + 4785],
                ],
                dim=1,
            )
            output = torch.cat([output, mon_output], dim=1)
        return output
