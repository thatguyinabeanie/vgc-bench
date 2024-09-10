import json
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Field,
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

from policy import MaskedActorCriticPolicy

with open("json/abilities.json") as f:
    ability_descs: dict[str, dict[str, npt.NDArray[np.float32]]] = json.load(f)
with open("json/items.json") as f:
    item_descs: dict[str, dict[str, npt.NDArray[np.float32]]] = json.load(f)
with open("json/moves.json") as f:
    move_descs: dict[str, dict[str, npt.NDArray[np.float32]]] = json.load(f)


class Agent(Player):
    __policy: MaskedActorCriticPolicy
    obs_len: int = 3380

    def __init__(
        self,
        policy: MaskedActorCriticPolicy | None,
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

    def set_policy(self, policy: MaskedActorCriticPolicy):
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
            lead_id = int(action.item()) + 1
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
            elif action < 6:
                return Player.create_order(Agent.active_first(list(battle.team.values()))[action])
            else:
                assert battle.active_pokemon is not None
                return Player.create_order(
                    list(battle.active_pokemon.moves.values())[(action - 6) % 4],
                    mega=10 <= action < 14,
                    z_move=14 <= action < 18,
                    dynamax=18 <= action < 22,
                    terastallize=22 <= action < 26,
                )
        else:
            return Player.choose_random_move(battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            mask = [float(i not in Agent.get_action_space(battle)) for i in range(26)]
            weather = [
                min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
                for w in Weather
            ]
            fields = [
                min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0
                for f in Field
            ]
            side_conditions = [
                (
                    0
                    if s not in battle.side_conditions
                    else (
                        1
                        if s == SideCondition.STEALTH_ROCK
                        else (
                            battle.side_conditions[s] / 2
                            if s == SideCondition.TOXIC_SPIKES
                            else (
                                battle.side_conditions[s] / 3
                                if s == SideCondition.SPIKES
                                else min(battle.turn - battle.side_conditions[s], 8) / 8
                            )
                        )
                    )
                )
                for s in SideCondition
            ]
            opp_side_conditions = [
                (
                    0
                    if s not in battle.opponent_side_conditions
                    else (
                        1
                        if s == SideCondition.STEALTH_ROCK
                        else (
                            battle.opponent_side_conditions[s] / 2
                            if s == SideCondition.TOXIC_SPIKES
                            else (
                                battle.opponent_side_conditions[s] / 3
                                if s == SideCondition.SPIKES
                                else min(battle.turn - battle.opponent_side_conditions[s], 8) / 8
                            )
                        )
                    )
                )
                for s in SideCondition
            ]
            if battle.active_pokemon is None:
                boosts = np.zeros(7)
                # effects = np.zeros(len(Effect))
                first_turn = 0
                protect_counter = 0
                must_recharge = 0
                preparing = 0
            else:
                boosts = [b / 6 for b in battle.active_pokemon.boosts.values()]
                # effects = [
                #     (
                #         min(battle.active_pokemon.effects[e], 8) / 8
                #         if e in battle.active_pokemon.effects
                #         else 0
                #     )
                #     for e in Effect
                # ]
                first_turn = float(battle.active_pokemon.first_turn)
                protect_counter = battle.active_pokemon.protect_counter / 5
                must_recharge = float(battle.active_pokemon.must_recharge)
                preparing = float(battle.active_pokemon.preparing)
            if battle.opponent_active_pokemon is None:
                opp_boosts = np.zeros(7)
                # opp_effects = np.zeros(len(Effect))
                opp_first_turn = 0
                opp_protect_counter = 0
                opp_must_recharge = 0
                opp_preparing = 0
            else:
                opp_boosts = [b / 6 for b in battle.opponent_active_pokemon.boosts.values()]
                # opp_effects = [
                #     (
                #         min(battle.opponent_active_pokemon.effects[e], 8) / 8
                #         if e in battle.opponent_active_pokemon.effects
                #         else 0
                #     )
                #     for e in Effect
                # ]
                opp_first_turn = float(battle.opponent_active_pokemon.first_turn)
                opp_protect_counter = battle.opponent_active_pokemon.protect_counter / 5
                opp_must_recharge = float(battle.opponent_active_pokemon.must_recharge)
                opp_preparing = float(battle.opponent_active_pokemon.preparing)
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
            preview = float(battle.in_team_preview)
            team = [Agent.embed_pokemon(p) for p in Agent.active_first(list(battle.team.values()))]
            team = np.concatenate([*team, np.zeros(270 * (6 - len(team)))])
            opp_team = [
                Agent.embed_pokemon(p) for p in Agent.active_first(list(battle.team.values()))
            ]
            opp_team = np.concatenate([*opp_team, np.zeros(270 * (6 - len(opp_team)))])
            return np.array(
                [
                    *mask,
                    *weather,
                    *fields,
                    *side_conditions,
                    *opp_side_conditions,
                    *boosts,
                    *opp_boosts,
                    # *effects,
                    # *opp_effects,
                    first_turn,
                    opp_first_turn,
                    protect_counter,
                    opp_protect_counter,
                    must_recharge,
                    opp_must_recharge,
                    preparing,
                    opp_preparing,
                    *special,
                    *opp_special,
                    force_switch,
                    preview,
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
        ability_desc = ability_descs[pokemon.ability or "null"]
        item_desc = item_descs[pokemon.item or "null"]
        moves = [Agent.embed_move(m) for m in pokemon.moves.values()]
        moves = np.concatenate([*moves, np.zeros(46 * (4 - len(moves)))])
        types = [float(t in pokemon.types) for t in PokemonType]
        hp = pokemon.max_hp / 714
        stats = [(s or 0) / 255 for s in pokemon.stats.values()]
        hp_frac = pokemon.current_hp_fraction
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        status = [float(s == pokemon.status) for s in Status]
        status_counter = pokemon.status_counter / 16
        weight = pokemon.weight / 1000
        active = float(pokemon.active or False)
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        specials = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        return np.array(
            [
                *ability_desc,
                *item_desc,
                *moves,
                *types,
                hp,
                *stats,
                hp_frac,
                *gender,
                *status,
                status_counter,
                weight,
                active,
                *tera_type,
                *specials,
            ]
        )

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        desc = move_descs[move.id if move.id[:11] != "hiddenpower" else "hiddenpower"]
        power = move.base_power / 250
        acc = move.accuracy / 100
        category = [float(c == move.category) for c in MoveCategory]
        priority = (move.priority + 7) / 12
        crit_ratio = move.crit_ratio
        drain = move.drain
        force_switch = float(move.force_switch)
        recoil = move.recoil
        self_destruct = float(move.self_destruct is not None)
        self_switch = float(move.self_switch is not False)
        pp = move.max_pp / 64
        pp_frac = move.current_pp / move.max_pp
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array(
            [
                *desc,
                power,
                acc,
                *category,
                priority,
                crit_ratio,
                drain,
                force_switch,
                recoil,
                self_destruct,
                self_switch,
                pp,
                pp_frac,
                *move_type,
            ]
        )

    @staticmethod
    def get_action_space(battle: Battle) -> list[int]:
        switch_space = [
            i
            for i, pokemon in enumerate(Agent.active_first(list(battle.team.values())))
            if not battle.maybe_trapped
            and pokemon.species in [p.species for p in battle.available_switches]
        ]
        if battle.active_pokemon is None:
            return switch_space
        else:
            move_space = [
                i + 6
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
            return switch_space + move_space + mega_space + zmove_space + dynamax_space + tera_space

    @staticmethod
    def active_first(pokemons: list[Pokemon]) -> list[Pokemon]:
        actives = [p for p in pokemons if p.active]
        if not actives:
            return pokemons
        else:
            return actives + [p for p in pokemons if not p.active]
