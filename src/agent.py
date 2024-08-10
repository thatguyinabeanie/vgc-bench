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
    MoveCategory,
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
    __policy: BasePolicy
    obs_len: int = 3415

    def __init__(
        self,
        policy: BasePolicy | None,
        device: torch.device | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy is not None:
            self.__policy = policy.to(device)
        else:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32),
                action_space=Discrete(26),
                lr_schedule=lambda _: 1e-4,
            ).to(device)

    def set_policy(self, policy: BasePolicy):
        self.__policy = policy.to(self.__policy.device)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, Battle):
            with torch.no_grad():
                embedded_battle = torch.tensor(
                    self.embed_battle(battle), device=self.__policy.device
                ).view(1, -1)
                action, _, _ = self.__policy.forward(embedded_battle)
            return Agent.action_to_move(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise TypeError()

    def teampreview(self, battle: AbstractBattle) -> str:
        if isinstance(battle, Battle):
            with torch.no_grad():
                embedded_battle = torch.tensor(
                    self.embed_battle(battle), device=self.__policy.device
                ).view(1, -1)
                action, _, _ = self.__policy.forward(embedded_battle)
            lead_id = int(action.item()) - 19
            all_ids = "123456"
            return "/team " + str(lead_id) + all_ids[: lead_id - 1] + all_ids[lead_id:]
        elif isinstance(battle, DoubleBattle):
            return self.random_teampreview(battle)
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
                raise LookupError(f"{action} not in {action_space}")
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
            mask = np.array([float(i not in Agent.get_action_space(battle)) for i in range(26)])
            weather = Agent.embed_weather(battle)
            force_switch = float(battle.force_switch)
            side_conditions = Agent.embed_side_conditions(battle)
            if battle.active_pokemon is None:
                boost_bins = [0] * 91
                effects = np.zeros(len(Effect))
            else:
                boost_bins = np.concatenate(
                    [
                        [float(b == i) for i in range(-6, 7)]
                        for b in battle.active_pokemon.boosts.values()
                    ]
                )
                effects = [float(e in battle.active_pokemon.effects.keys()) for e in Effect]
            if battle.opponent_active_pokemon is None:
                opp_boost_bins = [0] * 91
                opp_effects = np.zeros(len(Effect))
            else:
                opp_boost_bins = np.concatenate(
                    [
                        [float(b == i) for i in range(-6, 7)]
                        for b in battle.opponent_active_pokemon.boosts.values()
                    ]
                )
                opp_effects = [
                    float(e in battle.opponent_active_pokemon.effects.keys()) for e in Effect
                ]
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
            team = [Agent.embed_pokemon(p, False) for p in battle.team.values()]
            team = np.concatenate([*team, np.zeros(221 * (6 - len(battle.team)))])
            opp_team = [Agent.embed_pokemon(p, True) for p in battle.opponent_team.values()]
            opp_team = np.concatenate([*opp_team, np.zeros(221 * (6 - len(battle.opponent_team)))])
            return np.array(
                [
                    *mask,
                    *weather,
                    force_switch,
                    1 - force_switch,
                    *side_conditions,
                    *boost_bins,
                    *opp_boost_bins,
                    *effects,
                    *opp_effects,
                    *special,
                    *opp_special,
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
    def embed_weather(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        sun = [
            float(
                Weather.SUNNYDAY in battle.weather.keys()
                and min(battle.turn - battle.weather[Weather.SUNNYDAY], 8) == i
            )
            for i in range(9)
        ]
        rain = [
            float(
                Weather.RAINDANCE in battle.weather.keys()
                and min(battle.turn - battle.weather[Weather.RAINDANCE], 8) == i
            )
            for i in range(9)
        ]
        hail = [
            float(
                Weather.HAIL in battle.weather.keys()
                and min(battle.turn - battle.weather[Weather.HAIL], 8) == i
            )
            for i in range(9)
        ]
        sand = [
            float(
                Weather.SANDSTORM in battle.weather.keys()
                and min(battle.turn - battle.weather[Weather.SANDSTORM], 8) == i
            )
            for i in range(9)
        ]
        no_weather = not battle.weather
        return np.array([*sun, *rain, *hail, *sand, no_weather])

    @staticmethod
    def embed_side_conditions(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        stealth_rocks = float(SideCondition.STEALTH_ROCK in battle.side_conditions)
        opp_stealth_rocks = float(SideCondition.STEALTH_ROCK in battle.opponent_side_conditions)
        spikes = [
            float(
                SideCondition.SPIKES in battle.side_conditions
                and battle.side_conditions[SideCondition.SPIKES] == i
            )
            for i in range(4)
        ]
        opp_spikes = [
            float(
                SideCondition.SPIKES in battle.opponent_side_conditions
                and battle.opponent_side_conditions[SideCondition.SPIKES] == i
            )
            for i in range(4)
        ]
        toxic_spikes = [
            float(
                SideCondition.TOXIC_SPIKES in battle.side_conditions
                and battle.side_conditions[SideCondition.TOXIC_SPIKES] == i
            )
            for i in range(3)
        ]
        opp_toxic_spikes = [
            float(
                SideCondition.TOXIC_SPIKES in battle.opponent_side_conditions
                and battle.opponent_side_conditions[SideCondition.TOXIC_SPIKES] == i
            )
            for i in range(3)
        ]
        reflect = [
            float(
                SideCondition.REFLECT in battle.side_conditions
                and battle.turn - battle.side_conditions[SideCondition.REFLECT] == i
            )
            for i in range(10)
        ]
        opp_reflect = [
            float(
                SideCondition.REFLECT in battle.opponent_side_conditions
                and battle.turn - battle.opponent_side_conditions[SideCondition.REFLECT] == i
            )
            for i in range(10)
        ]
        light_screen = [
            float(
                SideCondition.LIGHT_SCREEN in battle.side_conditions
                and battle.turn - battle.side_conditions[SideCondition.LIGHT_SCREEN] == i
            )
            for i in range(10)
        ]
        opp_light_screen = [
            float(
                SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions
                and battle.turn - battle.opponent_side_conditions[SideCondition.LIGHT_SCREEN] == i
            )
            for i in range(10)
        ]
        safeguard = [
            float(
                SideCondition.SAFEGUARD in battle.side_conditions
                and battle.turn - battle.side_conditions[SideCondition.SAFEGUARD] == i
            )
            for i in range(7)
        ]
        opp_safeguard = [
            float(
                SideCondition.SAFEGUARD in battle.opponent_side_conditions
                and battle.turn - battle.opponent_side_conditions[SideCondition.SAFEGUARD] == i
            )
            for i in range(7)
        ]
        return np.array(
            [
                stealth_rocks,
                1 - stealth_rocks,
                opp_stealth_rocks,
                1 - opp_stealth_rocks,
                *spikes,
                *opp_spikes,
                *toxic_spikes,
                *opp_toxic_spikes,
                *reflect,
                *opp_reflect,
                *light_screen,
                *opp_light_screen,
                *safeguard,
                *opp_safeguard,
            ]
        )

    @staticmethod
    def embed_pokemon(pokemon: Pokemon, is_opponent: bool) -> npt.NDArray[np.float32]:
        # pokemon_id = POKEDEX.index(pokemon.species)
        # ability_id = ABILITYDEX.index(pokemon.ability)
        # item_id = ITEMDEX.index(pokemon.item)
        # move_ids = [MOVEDEX.index(m.id) for m in pokemon.moves.values()]
        # move_ids += [0] * (4 - len(move_ids))
        # move_pps = [
        #     np.floor((m.current_pp / m.max_pp) ** (1 / 3)) / 4 for m in pokemon.moves.values()
        # ]
        # move_pps += [0] * (4 - len(move_pps))
        moves = [Agent.embed_move(m) for m in pokemon.moves.values()]
        moves = np.concatenate([*moves, np.zeros(26 * (4 - len(pokemon.moves)))])
        types = [float(t in pokemon.types) for t in PokemonType]
        hp_frac_bins = [float((i - 1) / 6 < pokemon.current_hp_fraction <= i / 6) for i in range(7)]
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        status = [float(s == pokemon.status) for s in Status]
        toxic_counter = [
            float(pokemon.status == Status.TOX and pokemon.status_counter == i) for i in range(21)
        ]
        sleep_counter = [
            float(pokemon.status == Status.SLP and pokemon.status_counter == i) for i in range(11)
        ]
        weight_bins = [float(round(np.log10(pokemon.weight)) == i) for i in range(5)]
        height_bins = [float(round(np.log10(pokemon.weight)) == i) for i in range(4)]
        first_turn = float(pokemon.first_turn)
        protect_counter = [float(pokemon.protect_counter == i) for i in range(6)]
        must_recharge = float(pokemon.must_recharge)
        preparing = float(pokemon.preparing)
        active = float(pokemon.active or False)
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        specials = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        return np.array(
            [
                # pokemon_id,
                # ability_id,
                # item_id,
                # *move_ids,
                # *move_pps,
                *moves,
                *types,
                *hp_frac_bins,
                *gender,
                active,
                *status,
                *toxic_counter,
                *sleep_counter,
                *weight_bins,
                *height_bins,
                first_turn,
                1 - first_turn,
                *protect_counter,
                must_recharge,
                1 - must_recharge,
                preparing,
                1 - preparing,
                active,
                1 - active,
                float(is_opponent),
                1 - float(is_opponent),
                *tera_type,
                *specials,
            ]
        )

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        power = move.base_power / 250
        acc = move.accuracy / 100
        category = [float(c == move.category) for c in MoveCategory]
        pp_frac = move.current_pp / move.max_pp
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array([power, acc, *category, pp_frac, *move_type])

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
