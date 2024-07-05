from typing import Any

import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

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
        mask = obs[:, :10]  # type: ignore
        mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)
        mask = torch.where(mask == 1, float("-inf"), mask)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions + mask)
        values = self.value_net(latent_vf)
        return distribution, values
