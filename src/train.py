import asyncio
import os
import time

import torch
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import Agent
from env import ShowdownEnv, ShowdownVecEnvWrapper

BATTLE_FORMAT = "gen4ou"
TEAM = """
Bronzong @ Lum Berry
Ability: Heatproof
EVs: 248 HP / 252 Atk / 4 Def / 4 SpD
Brave Nature
IVs: 0 Spe
- Stealth Rock
- Gyro Ball
- Earthquake
- Explosion

Gengar @ Life Orb
Ability: Levitate
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Shadow Ball
- Focus Blast
- Explosion
- Hidden Power [Fire]

Tyranitar (M) @ Choice Band
Ability: Sand Stream
EVs: 252 Atk / 44 SpD / 212 Spe
Adamant Nature
- Stone Edge
- Crunch
- Pursuit
- Superpower

Kingdra @ Choice Specs
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hydro Pump
- Draco Meteor
- Surf
- Dragon Pulse

Flygon (M) @ Choice Scarf
Ability: Levitate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- U-turn
- Outrage
- Earthquake
- Thunder Punch

Lucario @ Life Orb
Ability: Inner Focus
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Close Combat
- Swords Dance
- Bullet Punch
- Extreme Speed"""
TOTAL_TIMESTEPS = 100_000_000


class SaveAndReplaceOpponentCallback(BaseCallback):
    def __init__(self, save_freq: int, num_saved_timesteps: int, n_steps: int):
        super().__init__()
        self.save_freq = save_freq
        self.total_timesteps = num_saved_timesteps
        self.n_steps = n_steps
        self.eval_opponents: list[Player] = [
            RandomPlayer(battle_format=BATTLE_FORMAT, team=TEAM),
            MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, team=TEAM),
            SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM),
        ]

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        agent = Agent(self.model.policy, battle_format=BATTLE_FORMAT, team=TEAM)
        self.model.env.set_opponent(agent)  # type: ignore
        self.total_timesteps += self.n_steps
        if self.total_timesteps % self.save_freq == 0:
            results = asyncio.run(agent.battle_against_multi(self.eval_opponents, n_battles=100))
            print(f"{time.strftime("%H:%M:%S")} - {results}")
            self.model.save(f"output/saves/ppo_{self.total_timesteps}")
            print(f"Saved checkpoint ppo_{self.total_timesteps}.zip")


class MaskedActorCriticPolicy(ActorCriticPolicy):
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
        probs = torch.softmax(distribution.distribution.probs[0] + mask, dim=0)  # type: ignore
        actions = torch.multinomial(probs, 1)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


def train():
    # setup
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=TEAM)
    env = ShowdownEnv(opponent, battle_format=BATTLE_FORMAT, log_level=40, team=TEAM)
    wrapper_env = ShowdownVecEnvWrapper(DummyVecEnv([lambda: env]), env)
    ppo = PPO(MaskedActorCriticPolicy, wrapper_env, tensorboard_log="output/logs/ppo")
    num_saved_timesteps = 0
    if os.path.exists("output/saves") and len(os.listdir("output/saves")) > 0:
        files = os.listdir("output/saves")
        num_saved_timesteps = max([int(file[4:-4]) for file in files])
        ppo.set_parameters(f"output/saves/ppo_{num_saved_timesteps}.zip")
        print(f"Resuming ppo_{num_saved_timesteps}.zip run.")
    opponent = Agent(ppo.policy, battle_format=BATTLE_FORMAT, team=TEAM)
    ppo.env.set_opponent(opponent)  # type: ignore

    def calc_learning_rate(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        saved_progress = num_saved_timesteps / TOTAL_TIMESTEPS
        return 10**-4.23 / (8 * (progress + saved_progress) / (1 + saved_progress) + 1) ** 1.5

    ppo.learning_rate = calc_learning_rate
    # train
    callback = SaveAndReplaceOpponentCallback(102_400, num_saved_timesteps, ppo.n_steps)
    ppo = ppo.learn(
        TOTAL_TIMESTEPS - num_saved_timesteps, callback=callback, reset_num_timesteps=False
    )


if __name__ == "__main__":
    train()
