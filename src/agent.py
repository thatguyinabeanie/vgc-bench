from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Move,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.player import BattleOrder, ForfeitBattleOrder, Player
from stable_baselines3.common.policies import BasePolicy

from policy import MaskedActorCriticPolicy


class Agent(Player):
    policy: BasePolicy
    obs_len: int = 2022

    def __init__(self, policy: BasePolicy | None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if policy is not None:
            self.policy = policy
        else:
            self.policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32),
                action_space=Discrete(26),
                lr_schedule=lambda x: 1e-4,
            )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, Battle):
            with torch.no_grad():
                embedded_battle = torch.tensor(
                    self.embed_battle(battle), device=self.policy.device
                ).view(1, -1)
                action, _, _ = self.policy.forward(embedded_battle)
            return Agent.action_to_move(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise TypeError()

    @staticmethod
    def action_to_move(action: int, battle: AbstractBattle) -> BattleOrder:
        if action == -1:
            return ForfeitBattleOrder()
        elif isinstance(battle, Battle):
            action_space = Agent.get_action_space(battle)
            if not action_space:
                return Player.choose_random_move(battle)
            elif action not in action_space:
                raise LookupError()
            elif action < 20:
                assert battle.active_pokemon is not None
                return Player.create_order(
                    list(battle.active_pokemon.moves.values())[action % 4],
                    mega=4 <= action < 8,
                    z_move=8 <= action < 12,
                    dynamax=12 <= action < 16,
                    terastallize=16 <= action < 20,
                )
            else:
                return Player.create_order(list(battle.team.values())[action - 20])
        else:
            return Player.choose_random_move(battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            assert battle.active_pokemon is not None
            assert battle.opponent_active_pokemon is not None
            mask = np.array([float(i not in Agent.get_action_space(battle)) for i in range(26)])
            boosts = np.array([b / 12 + 0.5 for b in battle.active_pokemon.boosts.values()])
            opp_boosts = np.array(
                [b / 12 + 0.5 for b in battle.opponent_active_pokemon.boosts.values()]
            )
            side_condition = np.array(
                [float(s in battle.side_conditions.keys()) for s in SideCondition]
            )
            opp_side_condition = np.array(
                [float(s in battle.opponent_side_conditions.keys()) for s in SideCondition]
            )
            effects = np.array([float(e in battle.active_pokemon.effects.keys()) for e in Effect])
            opp_effects = np.array(
                [float(e in battle.opponent_active_pokemon.effects.keys()) for e in Effect]
            )
            weather = np.array([float(w in battle.weather.keys()) for w in Weather])
            force_switch = np.array([float(battle.force_switch)])
            team = [Agent.embed_pokemon(p) for p in battle.team.values()]
            team = np.concatenate([*team, np.zeros(124 * (6 - len(battle.team)))])
            opp_team = [Agent.embed_pokemon(p) for p in battle.opponent_team.values()]
            opp_team = np.concatenate([*opp_team, np.zeros(124 * (6 - len(battle.opponent_team)))])
            return np.concatenate(
                [
                    mask,
                    boosts,
                    opp_boosts,
                    side_condition,
                    opp_side_condition,
                    effects,
                    opp_effects,
                    weather,
                    force_switch,
                    team,
                    opp_team,
                ],
                dtype=np.float32,
            )
        elif isinstance(battle, DoubleBattle):
            return np.array([])
        else:
            raise TypeError()

    @staticmethod
    def embed_pokemon(pokemon: Pokemon) -> npt.NDArray[np.float32]:
        level = pokemon.level / 100
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        hp_frac = pokemon.current_hp_fraction
        status = [float(s == pokemon.status) for s in Status]
        types = [float(t in pokemon.types) for t in PokemonType]
        moves = [Agent.embed_move(m) for m in pokemon.moves.values()]
        moves = np.concatenate([*moves, np.zeros(23 * (4 - len(pokemon.moves)))])
        return np.array([level, *gender, hp_frac, *status, *types, *moves])

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        power = move.base_power / 250
        acc = move.accuracy / 100
        pp_frac = move.current_pp / move.max_pp
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array([power, acc, pp_frac, *move_type])

    @staticmethod
    def get_action_space(battle: Battle) -> list[int]:
        if battle.active_pokemon is None:
            return []
        else:
            move_space = [
                i
                for i, move in enumerate(battle.active_pokemon.moves.values())
                if move.id in [m.id for m in battle.available_moves]
            ]
            mega_space = [i + 4 for i in move_space if battle.can_mega_evolve]
            zmove_space = [
                i + 8
                for i, move in enumerate(battle.active_pokemon.moves.values())
                if move.id in [m.id for m in battle.active_pokemon.available_z_moves]
                and battle.can_z_move
            ]
            dynamax_space = [i + 12 for i in move_space if battle.can_dynamax]
            tera_space = [i + 16 for i in move_space if battle.can_tera]
            switch_space = [
                i + 20
                for i, pokemon in enumerate(battle.team.values())
                if pokemon.species in [p.species for p in battle.available_switches]
            ]
            return move_space + mega_space + zmove_space + dynamax_space + tera_space + switch_space
