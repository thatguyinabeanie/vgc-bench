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
from poke_env.player import BattleOrder, ForfeitBattleOrder, Player
from stable_baselines3.common.policies import ActorCriticPolicy

from data import ability_descs, item_descs, move_descs
from policy import MaskedActorCriticPolicy


class Agent(Player):
    __policy: ActorCriticPolicy
    obs_len: int = 6734

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
                return Player.create_order(list(battle.team.values())[action])
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
            glob_features = Agent.embed_global(battle)
            side = Agent.embed_side(
                battle.side_conditions,
                [
                    battle.can_mega_evolve,
                    battle.can_z_move,
                    battle.can_dynamax,
                    battle.can_tera is not None,
                ],
                battle.team,
                battle.turn,
            )
            side = np.concatenate([np.concatenate([glob_features, s]) for s in side])
            opp_side = Agent.embed_side(
                battle.opponent_side_conditions,
                [
                    battle.opponent_can_mega_evolve,
                    battle.opponent_can_z_move,
                    battle.opponent_can_dynamax,
                    battle.opponent_can_tera,
                ],
                battle.opponent_team,
                battle.turn,
                opp=True,
            )
            opp_side = [np.concatenate([glob_features, s]) for s in opp_side]
            opp_side = np.concatenate([*opp_side, np.zeros(559 * (6 - len(opp_side)))])
            return np.concatenate([mask, side, opp_side], dtype=np.float32)
        elif isinstance(battle, DoubleBattle):
            return np.array([])
        else:
            raise TypeError()

    @staticmethod
    def embed_global(battle: Battle) -> npt.NDArray[np.float32]:
        weather = [
            min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0 for f in Field
        ]
        force_switch = float(battle.force_switch)
        preview = float(battle.in_team_preview)
        return np.array([*weather, *fields, force_switch, preview])

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
        pokemons = [Agent.embed_pokemon(p, i, opp) for i, p in enumerate(team.values())]
        return [np.concatenate([side_conditions, gims, p]) for p in pokemons]

    @staticmethod
    def embed_pokemon(pokemon: Pokemon, pos: int, from_opponent: bool) -> npt.NDArray[np.float32]:
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
        active = float(pokemon.active or False)
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
                active,
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
    def get_action_space(battle: Battle) -> list[int]:
        switch_space = [
            i
            for i, pokemon in enumerate(battle.team.values())
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
