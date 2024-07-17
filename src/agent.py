import json
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete
from poke_env.data.gen_data import GenData
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
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

DATA = GenData(gen=9)
POKEDEX_DICT = DATA.pokedex
POKEDEX = POKEDEX_DICT.keys()
MOVEDEX = DATA.moves.keys()
with open("json/abilities.json") as f:
    ABILITIES_DICT = json.load(f)
with open("json/items.json") as f:
    ITEMS = json.load(f).keys()


class Agent(Player):
    policy: BasePolicy
    obs_len: int = 22_254

    def __init__(self, policy: BasePolicy | None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if policy is not None:
            self.policy = policy
        else:
            self.policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32),
                action_space=Discrete(26),
                lr_schedule=lambda _: 1e-4,
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
            if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
                boosts = [0] * 7
                unboosts = [0] * 7
                opp_boosts = [0] * 7
                opp_unboosts = [0] * 7
                effects = [0] * len(Effect)
                opp_effects = [0] * len(Effect)
            else:
                boosts = [max(0, b / 6) for b in battle.active_pokemon.boosts.values()]
                unboosts = [max(0, -b / 6) for b in battle.active_pokemon.boosts.values()]
                opp_boosts = [max(0, b / 6) for b in battle.opponent_active_pokemon.boosts.values()]
                opp_unboosts = [
                    max(0, -b / 6) for b in battle.opponent_active_pokemon.boosts.values()
                ]
                effects = [float(e in battle.active_pokemon.effects.keys()) for e in Effect]
                opp_effects = [
                    float(e in battle.opponent_active_pokemon.effects.keys()) for e in Effect
                ]
            mask = np.array([float(i not in Agent.get_action_space(battle)) for i in range(26)])
            side_condition = [float(s in battle.side_conditions.keys()) for s in SideCondition]
            opp_side_condition = [
                float(s in battle.opponent_side_conditions.keys()) for s in SideCondition
            ]
            weather = [float(w in battle.weather.keys()) for w in Weather]
            special = [
                float(s)
                for s in [
                    battle.can_mega_evolve,
                    battle.can_z_move,
                    battle.can_dynamax,
                    battle.can_tera is not None,
                ]
            ]
            opp_special = [
                float(s)
                for s in [
                    battle.opponent_can_mega_evolve,
                    battle.opponent_can_z_move,
                    battle.opponent_can_dynamax,
                    battle.opponent_can_tera,
                ]
            ]
            force_switch = float(battle.force_switch)
            species_names = [p.species for p in battle.team.values()]
            species_multi_hot = [
                0 if ident not in species_names else (species_names.index(ident) + 1) / 6
                for ident in POKEDEX
            ]
            # if species_names:
            #     assert any(species_multi_hot)
            opp_species_names = [p.species for p in battle.opponent_team.values()]
            opp_species_multi_hot = [
                0 if ident not in opp_species_names else (opp_species_names.index(ident) + 1) / 6
                for ident in POKEDEX
            ]
            # if opp_species_names:
            #     assert any(opp_species_multi_hot)
            team = [Agent.embed_pokemon(p) for p in battle.team.values()]
            team = np.concatenate([*team, np.zeros(1551 * (6 - len(battle.team)))])
            opp_team = [Agent.embed_pokemon(p) for p in battle.opponent_team.values()]
            opp_team = np.concatenate([*opp_team, np.zeros(1551 * (6 - len(battle.opponent_team)))])
            return np.array(
                [
                    *mask,
                    *boosts,
                    *unboosts,
                    *opp_boosts,
                    *opp_unboosts,
                    *side_condition,
                    *opp_side_condition,
                    *effects,
                    *opp_effects,
                    *weather,
                    *special,
                    *opp_special,
                    force_switch,
                    *species_multi_hot,
                    *opp_species_multi_hot,
                    *team,
                    *opp_team,
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
        active = float(pokemon.active or False)
        status = [float(s == pokemon.status) for s in Status]
        types = [float(t in pokemon.types) for t in PokemonType]
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        specials = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        moves_multi_hot = [
            0 if m not in pokemon.moves.keys() else (list(pokemon.moves.keys()).index(m) + 1) / 4
            for m in MOVEDEX
        ]
        # if pokemon.moves:
        #     assert any(moves_multi_hot)
        moves_pp_frac = [m.current_pp / m.max_pp for m in pokemon.moves.values()]
        moves_pp_frac += [0] * (4 - len(pokemon.moves))
        ability = [
            float(pokemon.ability is not None and a == ABILITIES_DICT[pokemon.ability]["name"])
            for a in POKEDEX_DICT[pokemon.species]["abilities"].values()
        ]
        ability += [0] * (3 - len(ability))
        # if pokemon.ability is not None:
        #     assert any(ability)
        item = [float(i == (pokemon.item or "")) for i in ITEMS]
        # if pokemon.item and pokemon.item != "unknown_item":
        #     assert any(item)
        return np.array(
            [
                level,
                *gender,
                hp_frac,
                active,
                *status,
                *types,
                *tera_type,
                *specials,
                *moves_multi_hot,
                *moves_pp_frac,
                *ability,
                *item,
            ]
        )

    @staticmethod
    def get_action_space(battle: Battle) -> list[int]:
        switch_space = [
            i + 20
            for i, pokemon in enumerate(battle.team.values())
            if pokemon.species in [p.species for p in battle.available_switches]
        ]
        if battle.active_pokemon is None:
            return switch_space
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
            return move_space + mega_space + zmove_space + dynamax_space + tera_space + switch_space
