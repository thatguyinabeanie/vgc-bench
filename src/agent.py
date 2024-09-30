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
from poke_env.player import (
    BattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    Player,
)
from stable_baselines3.common.policies import ActorCriticPolicy

from data import ability_descs, item_descs, move_descs
from policy import MaskedActorCriticPolicy


class Agent(Player):
    __policy: ActorCriticPolicy
    obs_len: int = 8941

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
        else:
            self.__policy = MaskedActorCriticPolicy(
                observation_space=Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32),
                action_space=Discrete(2209),
                lr_schedule=lambda _: 1e-4,
            ).to(device)

    def set_policy(self, policy: MaskedActorCriticPolicy):
        self.__policy = policy.to(self.__policy.device)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        with torch.no_grad():
            embedded_battle = torch.tensor(
                self.embed_battle(battle), device=self.__policy.device
            ).view(1, -1)
            action, _, _ = self.__policy.forward(embedded_battle)
        return Agent.action_to_move(int(action.int()), battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        if isinstance(battle, DoubleBattle):
            with torch.no_grad():
                embedded_battle = torch.tensor(
                    self.embed_battle(battle), device=self.__policy.device
                ).view(1, -1)
                action, _, _ = self.__policy.forward(embedded_battle)
            order_message = "/team "
            all_ids = [str(i) for i in range(1, 7)]
            order_message += all_ids.pop(action // 60)
            action %= 120
            order_message += all_ids.pop(action // 12)
            action %= 24
            order_message += all_ids.pop(action // 3)
            action %= 6
            order_message += all_ids.pop(action)
            return order_message
        elif isinstance(battle, Battle):
            return self.random_teampreview(battle)
        else:
            raise TypeError()

    @staticmethod
    def action_to_move(action: int, battle: AbstractBattle) -> BattleOrder:
        # if battle.opponent_team:
        #     print(list(battle.opponent_team.values())[0].moves.keys())
        if action == -1:
            return ForfeitBattleOrder()
        action1 = action // 47
        action2 = action % 47
        order1 = Agent.action_to_move_ind(action1, battle, 0)
        order2 = Agent.action_to_move_ind(action2, battle, 1)
        # print(order1)
        # print(order2)
        return DoubleBattleOrder(order1, order2)

    @staticmethod
    def action_to_move_ind(action: int, battle: AbstractBattle, pos: int):
        if isinstance(battle, DoubleBattle):
            action_space = Agent.get_action_space_ind(battle, pos)
            # print(pos, action, action_space)
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
                    move_target=(action - 7) % 5 - 2,
                )
            return order
        else:
            return Player.choose_random_move(battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, DoubleBattle):
            action_space = Agent.get_action_space(battle)
            mask = [float(i not in action_space) for i in range(2209)]
            glob_features = Agent.embed_global(battle)
            side = Agent.embed_side(
                battle.side_conditions,
                [
                    battle.can_mega_evolve[0],
                    battle.can_z_move[0],
                    battle.can_dynamax[0],
                    battle.can_tera[0] is not False,
                ],
                battle.team,
                battle.turn,
            )
            side = np.concatenate([np.concatenate([glob_features, s]) for s in side])
            opp_side = Agent.embed_side(
                battle.opponent_side_conditions,
                [
                    battle.opponent_can_mega_evolve[0],
                    battle.opponent_can_z_move[0],
                    battle.opponent_can_dynamax[0],
                    battle._opponent_can_terrastallize,
                ],
                battle.opponent_team,
                battle.turn,
                opp=True,
            )
            opp_side = [np.concatenate([glob_features, s]) for s in opp_side]
            opp_side = np.concatenate([*opp_side, np.zeros(561 * (6 - len(opp_side)))])
            return np.concatenate([mask, side, opp_side], dtype=np.float32)
        else:
            return np.array([])

    @staticmethod
    def embed_global(battle: DoubleBattle) -> npt.NDArray[np.float32]:
        weather = [
            min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0 for f in Field
        ]
        force_switch = [float(f) for f in battle.force_switch]
        preview = float(battle.in_team_preview)
        return np.array([*weather, *fields, *force_switch, preview])

    @staticmethod
    def embed_side(
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
        pokemons = [Agent.embed_pokemon(p, n, i, opp) for i, (n, p) in enumerate(team.items())]
        return [np.concatenate([side_conditions, gims, p]) for p in pokemons]

    @staticmethod
    def embed_pokemon(
        pokemon: Pokemon, ident: str, pos: int, from_opponent: bool
    ) -> npt.NDArray[np.float32]:
        # (mostly) stable fields
        ability_desc = ability_descs["null" if pokemon.ability is None else pokemon.ability]
        item_desc = item_descs["null" if pokemon.item is None else pokemon.item]
        moves = [Agent.embed_move(m) for m in pokemon.moves.values()]
        moves = np.concatenate([*moves, np.zeros(46 * (4 - len(moves)))])
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
        active_a = float((pokemon.active or False) and ident.split()[0] in ["p1a", "p2a"])
        active_b = float((pokemon.active or False) and ident.split()[0] in ["p1b", "p2b"])
        pos_onehot = [float(pos == i) for i in range(6)]
        return np.array(
            [
                *ability_desc,
                *item_desc,
                *moves,
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
                active_a,
                active_b,
                *pos_onehot,
                float(from_opponent),
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
    def get_action_space(battle: DoubleBattle) -> list[int]:
        action_space1 = Agent.get_action_space_ind(battle, 0)
        action_space2 = Agent.get_action_space_ind(battle, 1)
        action_spaces = [
            [
                47 * a1 + a2
                for a2 in action_space2
                if not (1 <= a1 < 7 and 1 <= a2 < 7 and a1 == a2) and not (a1 >= 27 and a2 >= 27)
            ]
            for a1 in action_space1
        ]
        action_space = [a for actions in action_spaces for a in actions]
        return action_space

    @staticmethod
    def get_action_space_ind(battle: DoubleBattle, pos: int) -> list[int]:
        if battle.teampreview:
            return list(range(360))
        switch_space = [
            i + 1
            for i, pokemon in enumerate(battle.team.values())
            if not battle.maybe_trapped[pos]
            and battle.force_switch != [[False, True], [True, False]][pos]
            and not (len(battle.available_switches[0]) == 1 and battle.force_switch == [True, True] and pos == 1)
            and pokemon.species in [p.species for p in battle.available_switches[pos]]
        ]
        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return switch_space or [0]
        else:
            move_spaces = [
                [7 + 5 * i + j + 2 for j in battle.get_possible_showdown_targets(move, active_mon)]
                for i, move in enumerate(active_mon.moves.values())
                if move.id in [m.id for m in battle.available_moves[pos]]
            ]
            move_space = [i for s in move_spaces for i in s]
            tera_space = [i + 20 for i in move_space if battle.can_tera[pos]]
            return (switch_space + move_space + tera_space) or [0]
