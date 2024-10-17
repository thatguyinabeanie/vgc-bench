from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.player import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder, Player
from stable_baselines3.common.policies import ActorCriticPolicy

from data import abilities, items, moves
from policy import MaskedActorCriticPolicy


class Agent(Player):
    __policy: ActorCriticPolicy
    singles_act_len: int = 26
    doubles_act_len: int = 48
    base_obs_len: int = 6672
    singles_obs_len: int = singles_act_len + base_obs_len
    doubles_obs_len: int = 2 * doubles_act_len + base_obs_len

    def __init__(
        self,
        policy: ActorCriticPolicy | None,
        device: torch.device | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy is not None:
            self.__policy = policy.to(device)
        elif self.format_is_doubles:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(Agent.doubles_obs_len,), dtype=np.float32),
                action_space=MultiDiscrete([Agent.doubles_act_len, Agent.doubles_act_len]),
                lr_schedule=lambda _: 1e-4,
                mask_len=2 * Agent.doubles_act_len,
            ).to(device)
        else:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(Agent.singles_obs_len,), dtype=np.float32),
                action_space=Discrete(Agent.singles_act_len),
                lr_schedule=lambda _: 1e-4,
                mask_len=Agent.singles_act_len,
            ).to(device)

    def set_policy(self, policy: MaskedActorCriticPolicy):
        self.__policy = policy.to(self.__policy.device)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        embedded_battle = torch.tensor(self.embed_battle(battle), device=self.__policy.device).view(
            1, -1
        )
        with torch.no_grad():
            action, _, _ = self.__policy.forward(embedded_battle)
        if isinstance(battle, Battle):
            return Agent.singles_action_to_move(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            [action1, action2, *_] = action.cpu().numpy()[0]
            return Agent.doubles_action_to_move(action1, action2, battle)
        else:
            raise TypeError()

    def teampreview(self, battle: AbstractBattle) -> str:
        embedded_battle = torch.tensor(self.embed_battle(battle), device=self.__policy.device).view(
            1, -1
        )
        with torch.no_grad():
            action, _, _ = self.__policy.forward(embedded_battle)
        if isinstance(battle, Battle):
            order = Agent.singles_action_to_move_covering_teampreview(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            [action1, action2, *_] = action.cpu().numpy()[0]
            order = Agent.doubles_action_to_move_covering_teampreview(action1, action2, battle)
        else:
            raise TypeError()
        assert isinstance(order, str)
        return order

    @staticmethod
    def singles_action_to_move_covering_teampreview(
        action: int, battle: Battle
    ) -> BattleOrder | str:
        if battle.teampreview:
            return "/team 123456"
        else:
            return Agent.singles_action_to_move(action, battle)

    @staticmethod
    def singles_action_to_move(action: int, battle: Battle) -> BattleOrder:
        action_space = Agent.get_action_space(battle)
        if action == -1:
            return ForfeitBattleOrder()
        elif action not in action_space:
            raise LookupError(f"{action} not in {action_space}")
        elif action < 6:
            order = Player.create_order(list(battle.team.values())[action])
        else:
            active_mon = battle.active_pokemon
            assert active_mon is not None
            order = Player.create_order(
                list(active_mon.moves.values())[(action - 6) % 4],
                mega=10 <= action < 14,
                z_move=14 <= action < 18,
                dynamax=18 <= action < 22,
                terastallize=22 <= action < 26,
            )
        return order

    @staticmethod
    def doubles_action_to_move_covering_teampreview(
        action1: int, action2: int, battle: DoubleBattle
    ) -> BattleOrder | str:
        if battle.teampreview:
            assert action1 < 48 and action2 < 15
            all_ids = [str(i) for i in range(1, 7)]
            choices = ""
            choices += all_ids.pop(action1 // 8)
            action1 %= 8
            choices += all_ids.pop(action2 // 3)
            action2 %= 3
            choices += all_ids.pop(action1 // 2)
            action1 %= 2
            choices += all_ids.pop(action2)
            choices += all_ids.pop(action1)
            choices += all_ids.pop(0)
            order_message = f"/team {choices}"
            return order_message
        else:
            return Agent.doubles_action_to_move(action1, action2, battle)

    @staticmethod
    def doubles_action_to_move(action1: int, action2: int, battle: DoubleBattle) -> BattleOrder:
        if action1 == -1 or action2 == -1:
            return ForfeitBattleOrder()
        order1 = Agent.action_to_move_ind(action1, battle, 0)
        order2 = Agent.action_to_move_ind(action2, battle, 1)
        return DoubleBattleOrder(order1, order2)

    @staticmethod
    def action_to_move_ind(action: int, battle: DoubleBattle, pos: int) -> BattleOrder | None:
        action_space = Agent.get_action_space(battle, pos)
        if action not in action_space:
            raise LookupError(f"{action} not in {action_space}")
        elif action == 0:
            order = None
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            order = Player.create_order(
                list(active_mon.moves.values())[(action - 7) % 20 // 5],
                terastallize=bool((action - 7) // 20),
                move_target=(int(action) - 7) % 5 - 2,
            )
        return order

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            action_space = Agent.get_action_space(battle)
            mask = [float(i not in action_space) for i in range(Agent.singles_act_len)]
            actives = [] if battle.active_pokemon is None else [battle.active_pokemon.name]
            opp_actives = (
                []
                if battle.opponent_active_pokemon is None
                else [battle.opponent_active_pokemon.name]
            )
            gimmicks = [
                battle.can_mega_evolve,
                battle.can_z_move,
                battle.can_dynamax,
                battle.can_tera is not False,
            ]
            opp_gimmicks = [
                battle.opponent_can_mega_evolve,
                battle.opponent_can_z_move,
                battle.opponent_can_dynamax,
                battle._opponent_can_terrastallize,
            ]
        elif isinstance(battle, DoubleBattle):
            action_space1 = Agent.get_action_space(battle, 0)
            mask1 = [float(i not in action_space1) for i in range(Agent.doubles_act_len)]
            action_space2 = Agent.get_action_space(battle, 1)
            mask2 = [float(i not in action_space2) for i in range(Agent.doubles_act_len)]
            mask = mask1 + mask2
            actives = [p.name for p in battle.active_pokemon if p is not None]
            opp_actives = [p.name for p in battle.opponent_active_pokemon if p is not None]
            gimmicks = [
                battle.can_mega_evolve[0],
                battle.can_z_move[0],
                battle.can_dynamax[0],
                battle.can_tera[0] is not False,
            ]
            opp_gimmicks = [
                battle.opponent_can_mega_evolve[0],
                battle.opponent_can_z_move[0],
                battle.opponent_can_dynamax[0],
                battle._opponent_can_terrastallize,
            ]
        else:
            raise TypeError()
        glob_features = Agent.embed_global(battle)
        side = Agent.embed_side(actives, battle.side_conditions, gimmicks, battle.team, battle.turn)
        side = np.concatenate([np.concatenate([glob_features, s]) for s in side])
        opp_side = Agent.embed_side(
            opp_actives,
            battle.opponent_side_conditions,
            opp_gimmicks,
            battle.opponent_team,
            battle.turn,
            opp=True,
        )
        opp_side = [np.concatenate([glob_features, s]) for s in opp_side]
        opp_side = np.concatenate([*opp_side, np.zeros(556 * (6 - len(opp_side)))])
        return np.concatenate([mask, side, opp_side], dtype=np.float32)

    @staticmethod
    def embed_global(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        weather = [
            min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0 for f in Field
        ]
        preview = float(battle.in_team_preview)
        if isinstance(battle, Battle):
            force_switch = [float(battle.force_switch), 0]
        elif isinstance(battle, DoubleBattle):
            force_switch = [float(f) for f in battle.force_switch]
        else:
            raise TypeError()
        return np.array([preview, *weather, *fields, *force_switch])

    @staticmethod
    def embed_side(
        actives: list[str],
        side_conds: dict[SideCondition, int],
        gimmicks: list[bool],
        team: dict[str, Pokemon],
        turn: int,
        opp: bool = False,
    ) -> list[npt.NDArray[np.float32]]:
        side_conditions = [
            (
                0
                if s not in side_conds
                else (
                    1
                    if s == SideCondition.STEALTH_ROCK
                    else (
                        side_conds[s] / 2
                        if s == SideCondition.TOXIC_SPIKES
                        else (
                            side_conds[s] / 3
                            if s == SideCondition.SPIKES
                            else min(turn - side_conds[s], 8) / 8
                        )
                    )
                )
            )
            for s in SideCondition
        ]
        gims = [float(g) for g in gimmicks]
        pokemons = [
            Agent.embed_pokemon(
                p,
                i,
                opp,
                len(actives) > 0 and p.name == actives[0],
                len(actives) > 1 and p.name == actives[1],
            )
            for i, p in enumerate(team.values())
        ]
        return [np.concatenate([side_conditions, gims, p]) for p in pokemons]

    @staticmethod
    def embed_pokemon(
        pokemon: Pokemon, pos: int, from_opponent: bool, active_a: bool, active_b: bool
    ) -> npt.NDArray[np.float32]:
        # (mostly) stable fields
        ability_id = abilities.index("null" if pokemon.ability is None else pokemon.ability)
        item_id = items.index("null" if pokemon.item is None else pokemon.item)
        move_ids = [
            moves.index(m.id if m.id[:11] != "hiddenpower" else "hiddenpower")
            for m in pokemon.moves.values()
        ]
        move_ids = move_ids + [0] * (4 - len(move_ids))
        move_details = [Agent.embed_move(m) for m in pokemon.moves.values()]
        move_details = np.concatenate([*move_details, np.zeros(49 * (4 - len(move_details)))])
        types = [float(t in pokemon.types) for t in PokemonType]
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        stats = [(s or 0) / 1000 for s in pokemon.stats.values()]
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        weight = pokemon.weight / 1000
        # volatile fields
        hp_frac = pokemon.current_hp_fraction
        status = [float(s == pokemon.status) for s in Status]
        status_counter = pokemon.status_counter / 16
        boosts = [b / 6 for b in pokemon.boosts.values()]
        effects = [(min(pokemon.effects[e], 8) / 8 if e in pokemon.effects else 0) for e in Effect]
        first_turn = float(pokemon.first_turn)
        protect_counter = pokemon.protect_counter / 5
        must_recharge = float(pokemon.must_recharge)
        preparing = float(pokemon.preparing)
        gimmicks = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        pos_onehot = [float(pos == i) for i in range(6)]
        return np.array(
            [
                ability_id,
                item_id,
                *move_ids,
                *move_details,
                *types,
                *tera_type,
                *stats,
                *gender,
                weight,
                hp_frac,
                *status,
                status_counter,
                *boosts,
                *effects,
                first_turn,
                protect_counter,
                must_recharge,
                preparing,
                *gimmicks,
                float(active_a),
                float(active_b),
                *pos_onehot,
                float(from_opponent),
            ]
        )

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        power = move.base_power / 250
        acc = move.accuracy / 100
        category = [float(c == move.category) for c in MoveCategory]
        target = [float(t == move.target) for t in Target]
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
                power,
                acc,
                *category,
                *target,
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
    def get_action_space(battle: AbstractBattle, pos: int | None = None) -> npt.NDArray[np.int64]:
        if isinstance(battle, Battle):
            switch_space = [
                i
                for i, pokemon in enumerate(battle.team.values())
                if not battle.maybe_trapped
                and pokemon.species in [p.species for p in battle.available_switches]
            ]
            if battle.active_pokemon is None:
                return np.array(switch_space)
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
                return np.array(
                    switch_space
                    + move_space
                    + mega_space
                    + zmove_space
                    + dynamax_space
                    + tera_space
                )
        elif isinstance(battle, DoubleBattle):
            assert pos is not None
            if battle.finished:
                return np.array([])
            if battle.teampreview:
                return np.array(range(48) if pos == 0 else range(15))
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if not battle.maybe_trapped[pos]
                and battle.force_switch != [[False, True], [True, False]][pos]
                and not (
                    len(battle.available_switches[0]) == 1
                    and battle.force_switch == [True, True]
                    and pos == 1
                )
                and pokemon.species in [p.species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            if active_mon is None:
                return np.array(switch_space or [0])
            else:
                move_spaces = [
                    [
                        7 + 5 * i + j + 2
                        for j in battle.get_possible_showdown_targets(move, active_mon)
                    ]
                    for i, move in enumerate(active_mon.moves.values())
                    if move.id in [m.id for m in battle.available_moves[pos]]
                ]
                move_space = [i for s in move_spaces for i in s]
                tera_space = [i + 20 for i in move_space if battle.can_tera[pos]]
                if (
                    not move_space
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    move_space = [9]
                return np.array((switch_space + move_space + tera_space) or [0])
        else:
            raise TypeError()
