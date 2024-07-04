import asyncio
import time
from typing import Any

import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs

from agent import Agent


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = ~obs[:, :10].bool()
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        masked_actions = (
            mean_actions if mask.all() else mean_actions.masked_fill(mask, float("-inf"))
        )
        distribution = self.action_dist.proba_distribution(action_logits=masked_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        print("actions:", actions)
        mask = ~obs[:, :10].bool()  # type: ignore
        print("mask:", mask)
        print(mask[torch.arange(mask.size(0)), actions])
        features = self.extract_features(obs)
        # print("features:", features)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)  # THE PROBLEM IS HERE
        else:  # ignore this, it is skipped every time
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # print("latent_pi:", latent_pi)
        # for name, param in self.mlp_extractor.named_parameters():
        #     if param._grad is not None:
        #         print(name, "grad:", (param._grad**2).mean())
        mean_actions = self.action_net(latent_pi)
        masked_actions = torch.zeros(mean_actions.size()).to(self.device)
        for i in range(mask.size(0)):
            masked_actions[i] = (
                mean_actions[i]
                if mask[i].all()
                else mean_actions[i].masked_fill(mask[i], float("-inf"))
            )
        distribution = self.action_dist.proba_distribution(action_logits=masked_actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


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
            RandomPlayer(battle_format=self.battle_format, team=eval_team),
            MaxBasePowerPlayer(battle_format=self.battle_format, team=eval_team),
            SimpleHeuristicsPlayer(battle_format=self.battle_format, team=eval_team),
        ]

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        self.total_timesteps += self.n_steps
        if self.self_play:
            agent = Agent(self.model.policy, battle_format=self.battle_format, team=self.team)
            self.model.env.set_opponent(agent)  # type: ignore
        if self.total_timesteps % self.save_freq == 0:
            agent = Agent(self.model.policy, battle_format=self.battle_format, team=self.team)
            results = asyncio.run(agent.battle_against_multi(self.eval_opponents, n_battles=100))
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")
