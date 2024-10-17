from __future__ import annotations

from typing import Any

import torch
from gymnasium import Space
from gymnasium.spaces import Discrete
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import Distribution, MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from data import abilities, items, moves


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, mask_len: int, **kwargs: Any):
        self.mask_len = mask_len
        super().__init__(
            *args,
            **kwargs,
            net_arch=[],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=AttentionExtractor,
            features_extractor_kwargs={"mask_len": mask_len},
            share_features_extractor=False,
            # optimizer_kwargs={"weight_decay": 1e-5},
        )

    @classmethod
    def clone(cls, model: BaseAlgorithm) -> MaskedActorCriticPolicy:
        new_policy = cls(
            model.observation_space,
            model.action_space,
            model.lr_schedule,
            mask_len=model.policy.mask_len,
        )
        new_policy.load_state_dict(model.policy.state_dict())
        return new_policy

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, value_logits = self.get_logits(obs)
        distribution = self.get_dist_from_logits(obs, action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        if isinstance(distribution, MultiCategoricalDistribution):
            distribution2 = self.get_dist_from_logits(obs, action_logits, actions[:, :1])
            assert isinstance(distribution2, MultiCategoricalDistribution)
            actions2 = distribution2.get_actions(deterministic=deterministic)
            distribution.distribution[1] = distribution2.distribution[1]
            actions[:, 1] = actions2[:, 1]
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, value_logits, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert isinstance(obs, torch.Tensor)
        action_logits, value_logits = self.get_logits(obs)
        distribution = self.get_dist_from_logits(obs, action_logits)
        if isinstance(distribution, MultiCategoricalDistribution):
            distribution2 = self.get_dist_from_logits(obs, action_logits, actions[:, :1])
            assert isinstance(distribution2, MultiCategoricalDistribution)
            distribution.distribution[1] = distribution2.distribution[1]
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value_logits, log_prob, entropy

    def get_logits(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)  # type: ignore
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        action_logits = self.action_net(latent_pi)
        value_logits = self.value_net(latent_vf)
        return action_logits, value_logits

    def get_dist_from_logits(
        self, obs: torch.Tensor, action_logits: torch.Tensor, action: torch.Tensor | None = None
    ) -> Distribution:
        mask = self.get_mask(obs, action)
        distribution = self.action_dist.proba_distribution(action_logits + mask)
        return distribution

    def get_mask(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.action_space, Discrete):
            mask = obs[:, : self.action_space.n]  # type: ignore
            mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)
            mask = torch.where(mask == 1, float("-inf"), mask)
            return mask
        else:
            act_len = self.action_space.nvec[0]  # type: ignore
            if action is None:
                mask = obs[:, : 2 * act_len]
            else:
                mask = obs[:, act_len : 2 * act_len]
                # condition: not in teampreview and action is either switch or terastallization
                condition = (obs[:, 2 * act_len].view(-1, 1) == 0) & (
                    ((1 <= action) & (action < 7)) | (action >= 27)
                )
                indices = torch.arange(act_len, device=mask.device).unsqueeze(0).expand_as(mask)
                action_mask = indices == action
                mask = torch.where(condition & action_mask, 1.0, mask)
                mask = torch.cat([obs[:, :act_len], mask], dim=1)
            mask = torch.where(mask == 1, float("-inf"), mask)
            return mask


class AttentionExtractor(BaseFeaturesExtractor):
    chunk_len: int = 555
    embed_len: int = 50
    feature_len: int = 128

    def __init__(self, observation_space: Space[Any], mask_len: int):
        super().__init__(observation_space, features_dim=self.feature_len)
        self.mask_len = mask_len
        self.ability_embedding = nn.Embedding(len(abilities), self.embed_len)
        self.item_embedding = nn.Embedding(len(items), self.embed_len)
        self.move_embedding = nn.Embedding(len(moves), self.embed_len)
        self.feature_proj = nn.Linear(self.chunk_len + 6 * (self.embed_len - 1), self.feature_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_len))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_len,
                nhead=1,
                dim_feedforward=self.feature_len,
                dropout=0,
                batch_first=True,
            ),
            num_layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, self.mask_len :].view(-1, 12, self.chunk_len)
        seq = self.embed(seq)
        seq = self.feature_proj(seq)
        seq = torch.cat([self.cls_token.expand(seq.size(0), -1, -1), seq], dim=1)
        return self.encoder(seq)[:, 0, :]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        ability = self.ability_embedding(x[:, :, 0].long())
        item = self.item_embedding(x[:, :, 1].long())
        move = self.move_embedding(x[:, :, 2:6].long()).view(-1, 12, 4 * self.embed_len)
        return torch.cat([ability, item, move, x[:, :, 6:]], dim=2)
