from __future__ import annotations

from typing import Any

import torch
from gymnasium import Space
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=AttentionExtractor,
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
        mask = obs[:, : self.action_space.n]  # type: ignore
        mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)  # type: ignore
        mask = torch.where(mask == 1, float("-inf"), mask)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions + mask)
        values = self.value_net(latent_vf)
        return distribution, values


class AttentionExtractor(BaseFeaturesExtractor):
    battle_len: int = 140
    pokemon_len: int = 270
    feature_len: int = 128

    def __init__(self, observation_space: Space[Any]):
        super().__init__(observation_space, features_dim=self.feature_len)
        self.chunk_sizes = [self.battle_len] + [self.pokemon_len] * 12
        self.feature_proj = nn.Linear(self.battle_len + self.pokemon_len, self.feature_len)
        self.register_buffer("positions", torch.arange(12).unsqueeze(0))
        self.pos_embedding = nn.Embedding(12, self.feature_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_len,
            nhead=4,
            dim_feedforward=self.feature_len,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        battle, *pokemons = torch.split(x, self.chunk_sizes, dim=1)
        seq = torch.stack([torch.cat([battle, p], dim=1) for p in pokemons], dim=1)
        seq = self.feature_proj.forward(seq)
        seq += self.pos_embedding.forward(self.positions)
        output = self.encoder.forward(seq)
        return output[:, 0, :]
