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
            net_arch=[512, 256, 128, 64],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=CrossAttentionReducer,
            # optimizer_kwargs={"weight_decay": 1e-4},
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


class CrossAttentionReducer(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space[Any]):
        super().__init__(observation_space, features_dim=1024)
        self.query_layer = torch.nn.Linear(564, 1024)
        self.key_layer = torch.nn.Linear(392, 1024)
        self.value_layer = torch.nn.Linear(392, 1024)
        self.chunk_sizes = [self.query_layer.in_features] + [self.key_layer.in_features] * 12
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=1024, num_heads=16, batch_first=True  # , dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.chunk_sizes, dim=1)
        query = self.query_layer(chunks[0]).unsqueeze(1)
        key = torch.stack([self.key_layer(c) for c in chunks[1:]], dim=1)
        value = torch.stack([self.value_layer(c) for c in chunks[1:]], dim=1)
        output, _ = self.cross_attention(query=query, key=key, value=value)
        return output.squeeze(1)
