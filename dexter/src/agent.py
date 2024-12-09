from copy import deepcopy
from typing import Any, Deque

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
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.player import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder, Player
from src.policy import MaskedActorCriticPolicy
from src.utils import (
    abilities,
    doubles_act_len,
    doubles_chunk_obs_len,
    items,
    moves,
    pokemon_obs_len,
    singles_act_len,
    singles_chunk_obs_len,
)
from stable_baselines3.common.policies import ActorCriticPolicy


class Agent(Player):
    __policy: ActorCriticPolicy
    frames: Deque[AbstractBattle]
    __teampreview_draft: list[str]

    def __init__(
        self,
        policy: ActorCriticPolicy | None,
        num_frames: int,
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
                observation_space=Box(
                    -1, len(moves), shape=(12, doubles_chunk_obs_len), dtype=np.float32
                ),
                action_space=MultiDiscrete([doubles_act_len, doubles_act_len]),
                lr_schedule=lambda _: 1e-5,
                num_frames=num_frames,
            ).to(device)
        else:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(
                    -1, len(moves), shape=(12, singles_chunk_obs_len), dtype=np.float32
                ),
                action_space=Discrete(singles_act_len),
                lr_schedule=lambda _: 1e-5,
                num_frames=num_frames,
            ).to(device)
        self.frames = Deque(maxlen=num_frames)
        self.__teampreview_draft = []

    def set_policy(self, policy: MaskedActorCriticPolicy):
        self.__policy = policy.to(self.__policy.device)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.frames.append(battle)
        obs = np.stack(
            [self.embed_battle(frame, self.__teampreview_draft) for frame in self.frames]
        )
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.__policy.device).unsqueeze(0)
            action, _, _ = self.__policy.forward(obs_tensor)
        if isinstance(battle, Battle):
            return Agent.singles_action_to_move(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            [action1, action2, *_] = action.cpu().numpy()[0]
            return Agent.doubles_action_to_move(action1, action2, battle)
        else:
            raise TypeError()

    def teampreview(self, battle: AbstractBattle) -> str:
        if isinstance(battle, Battle):
            return self.random_teampreview(battle)
        elif isinstance(battle, DoubleBattle):
            order1 = self.choose_move(battle)
            assert isinstance(order1, DoubleBattleOrder)
            pokemon1 = None if order1.first_order is None else order1.first_order.order
            pokemon2 = None if order1.second_order is None else order1.second_order.order
            assert isinstance(pokemon1, Pokemon)
            assert isinstance(pokemon2, Pokemon)
            self.__teampreview_draft = [pokemon1.name, pokemon2.name]
            action1 = [p.name for p in battle.team.values()].index(pokemon1.name) + 1
            action2 = [p.name for p in battle.team.values()].index(pokemon2.name) + 1
            battle2 = deepcopy(battle)
            battle2.switch(
                f"{battle2.player_role}a: {pokemon1.base_species.capitalize()}",
                "",
                f"{pokemon1.current_hp}/{pokemon1.max_hp}",
            )
            battle2.switch(
                f"{battle2.player_role}b: {pokemon2.base_species.capitalize()}",
                "",
                f"{pokemon2.current_hp}/{pokemon2.max_hp}",
            )
            order2 = self.choose_move(battle2)
            assert isinstance(order2, DoubleBattleOrder)
            pokemon3 = None if order2.first_order is None else order2.first_order.order
            pokemon4 = None if order2.second_order is None else order2.second_order.order
            assert isinstance(pokemon3, Pokemon)
            assert isinstance(pokemon4, Pokemon)
            self.__teampreview_draft += [pokemon3.name, pokemon4.name]
            action3 = [p.name for p in battle2.team.values()].index(pokemon3.name) + 1
            action4 = [p.name for p in battle2.team.values()].index(pokemon4.name) + 1
            return f"/team {action1}{action2}{action3}{action4}"
        else:
            raise TypeError()

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
    def doubles_action_to_move(action1: int, action2: int, battle: DoubleBattle) -> BattleOrder:
        if action1 == -1 or action2 == -1:
            return ForfeitBattleOrder()
        order1 = Agent.action_to_move_ind(action1, battle, 0)
        order2 = Agent.action_to_move_ind(action2, battle, 1)
        return DoubleBattleOrder(order1, order2)

    @staticmethod
    def action_to_move_ind(action: int, battle: DoubleBattle, pos: int) -> BattleOrder | None:
        action_space = Agent.get_action_space(battle, pos)
        assert action in action_space, f"{action} not in {action_space}"
        if action == 0:
            order = None
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            mvs = (
                battle.available_moves[pos]
                if len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                else list(active_mon.moves.values())
            )
            order = Player.create_order(
                mvs[(action - 7) % 20 // 5],
                terastallize=bool((action - 7) // 20),
                move_target=(int(action) - 7) % 5 - 2,
            )
        return order

    @staticmethod
    def doubles_order_to_action(
        order: DoubleBattleOrder, battle: DoubleBattle
    ) -> npt.NDArray[np.int32]:
        action1 = Agent.order_to_action_ind(order.first_order, battle, pos=0)
        action2 = Agent.order_to_action_ind(order.second_order, battle, pos=1)
        return np.array([action1, action2])

    @staticmethod
    def order_to_action_ind(order: BattleOrder | None, battle: DoubleBattle, pos: int) -> int:
        if order is None:
            action = 0
        else:
            order_item = order.order
            if isinstance(order_item, Pokemon):
                index = [p.name for p in battle.team.values()].index(order_item.name)
                action = 1 + index
            elif isinstance(order_item, Move):
                active = battle.active_pokemon[pos]
                assert active is not None
                index = [m.id for m in active.moves.values()].index(order_item.id)
                tera = 20 if order.terastallize else 0
                target = order.move_target + 2
                action = 7 + 5 * index + target + tera
            else:
                raise ValueError()
        return action

    @staticmethod
    def embed_battle(
        battle: AbstractBattle, teampreview_draft: list[str]
    ) -> npt.NDArray[np.float32]:
        glob = Agent.embed_global(battle, teampreview_draft)
        side = Agent.embed_side(battle)
        opp_side = Agent.embed_side(battle, opp=True)
        [a1, a2, *_] = (
            [battle.active_pokemon] if isinstance(battle, Battle) else battle.active_pokemon
        )
        [o1, o2, *_] = (
            [battle.opponent_active_pokemon]
            if isinstance(battle, Battle)
            else battle.opponent_active_pokemon
        )
        pokemons = [
            Agent.embed_pokemon(
                p,
                i,
                from_opponent=False,
                active_a=a1 is not None and p.name == a1.name,
                active_b=a2 is not None and p.name == a2.name,
            )
            for i, p in enumerate(battle.team.values())
        ]
        pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (6 - len(pokemons))
        opp_pokemons = [
            Agent.embed_pokemon(
                p,
                i,
                from_opponent=True,
                active_a=o1 is not None and p.name == o1.name,
                active_b=o2 is not None and p.name == o2.name,
            )
            for i, p in enumerate(battle.opponent_team.values())
        ]
        opp_pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (6 - len(opp_pokemons))
        return np.stack(
            [np.concatenate([glob, side, p]) for p in pokemons]
            + [np.concatenate([glob, opp_side, p]) for p in opp_pokemons],
            dtype=np.float32,
        )

    @staticmethod
    def embed_global(
        battle: AbstractBattle, teampreview_draft: list[str]
    ) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            if not battle._last_request:
                mask = np.zeros(singles_act_len, dtype=np.float32)
            else:
                action_space = Agent.get_action_space(battle)
                mask = [float(i not in action_space) for i in range(singles_act_len)]
            force_switch = [float(battle.force_switch), 0]
        elif isinstance(battle, DoubleBattle):
            if not battle._last_request:
                mask = np.zeros(2 * doubles_act_len, dtype=np.float32)
            else:
                action_space1 = Agent.get_action_space(battle, 0)
                mask1 = [float(i not in action_space1) for i in range(doubles_act_len)]
                action_space2 = Agent.get_action_space(battle, 1)
                mask2 = [float(i not in action_space2) for i in range(doubles_act_len)]
                mask = mask1 + mask2
            force_switch = [float(f) for f in battle.force_switch]
        else:
            raise TypeError()
        weather = [
            min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0 for f in Field
        ]
        if battle.teampreview:
            if not battle.active_pokemon:
                teampreview_draft = []
            else:
                teampreview_draft = teampreview_draft[:2]
        draft_positions = [float(p.name in teampreview_draft) for p in battle.team.values()]
        return np.concatenate(
            [mask, weather, fields, draft_positions, force_switch], dtype=np.float32
        )

    @staticmethod
    def embed_side(battle: AbstractBattle, opp: bool = False) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            gims = [
                battle.can_mega_evolve,
                battle.can_z_move,
                battle.can_dynamax,
                battle.can_tera is not False,
            ]
            opp_gims = [
                battle.opponent_can_mega_evolve,
                battle.opponent_can_z_move,
                battle.opponent_can_dynamax,
                battle._opponent_can_terrastallize,
            ]
        elif isinstance(battle, DoubleBattle):
            gims = [
                battle.can_mega_evolve[0],
                battle.can_z_move[0],
                battle.can_dynamax[0],
                battle.can_tera[0] is not False,
            ]
            opp_gims = [
                battle.opponent_can_mega_evolve[0],
                battle.opponent_can_z_move[0],
                battle.opponent_can_dynamax[0],
                battle._opponent_can_terrastallize,
            ]
        else:
            raise TypeError()
        side_conds = battle.opponent_side_conditions if opp else battle.side_conditions
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
                            else min(battle.turn - side_conds[s], 8) / 8
                        )
                    )
                )
            )
            for s in SideCondition
        ]
        gims = opp_gims if opp else gims
        gimmicks = [float(g) for g in gims]
        return np.concatenate([side_conditions, gimmicks], dtype=np.float32)

    @staticmethod
    def embed_pokemon(
        pokemon: Pokemon, pos: int, from_opponent: bool, active_a: bool, active_b: bool
    ) -> npt.NDArray[np.float32]:
        # (mostly) stable fields
        ability_id = abilities.index("null" if pokemon.ability is None else pokemon.ability)
        item_id = items.index("null" if pokemon.item is None else pokemon.item)
        move_ids = [
            moves.index("hiddenpower" if move.id.startswith("hiddenpower") else move.id)
            for move in pokemon.moves.values()
        ]
        move_ids += [0] * (4 - len(move_ids))
        types = [float(t in pokemon.types) for t in PokemonType]
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        stats = [(s or 0) / 1000 for s in pokemon.stats.values()]
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        weight = pokemon.weight / 1000
        # volatile fields
        hp_frac = pokemon.current_hp_fraction
        pp_fracs = [m.current_pp / m.max_pp for m in pokemon.moves.values()]
        pp_fracs += [0] * (4 - len(pp_fracs))
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
                *types,
                *tera_type,
                *stats,
                *gender,
                weight,
                hp_frac,
                *pp_fracs,
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
            ],
            dtype=np.float32,
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
                and not pokemon.active
                and pokemon.species in [p.species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            if battle.teampreview:
                return np.array(switch_space)
            elif active_mon is None:
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
