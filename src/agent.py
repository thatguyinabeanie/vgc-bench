from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Move,
    Pokemon,
    PokemonType,
    Status,
)
from poke_env.player import BattleOrder, ForfeitBattleOrder, Player
from stable_baselines3.common.policies import BasePolicy


class Agent(Player):
    policy: BasePolicy

    def __init__(self, policy: BasePolicy, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy

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
            if not Agent.get_action_space(battle):
                return Player.choose_random_move(battle)
            elif action not in Agent.get_action_space(battle):
                return Player.choose_random_move(battle)
            elif action < 4:
                assert battle.active_pokemon is not None
                return Player.create_order(list(battle.active_pokemon.moves.values())[action])
            else:
                return Player.create_order(list(battle.team.values())[action - 4])
        else:
            return Player.choose_random_move(battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            return np.concatenate(
                [[float(i in Agent.get_action_space(battle)) for i in range(10)]]
                + [Agent.embed_pokemon(p) for p in battle.team.values()]
                + [Agent.embed_pokemon(p) for p in battle.opponent_team.values()]
                + [torch.zeros(117)] * (12 - len(battle.team) - len(battle.opponent_team)),
                dtype=np.float32,
            )
        elif isinstance(battle, DoubleBattle):
            return np.array([])
        else:
            raise TypeError()

    @staticmethod
    def embed_pokemon(pokemon: Pokemon) -> npt.NDArray[np.float32]:
        level = pokemon.level / 100
        hp_frac = pokemon.current_hp_fraction
        status = [float(s == pokemon.status) for s in Status]
        types = [float(t in pokemon.types) for t in PokemonType]
        moves = [Agent.embed_move(m) for m in pokemon.moves.values()] + [torch.zeros(22)] * (
            4 - len(pokemon.moves)
        )
        return np.concatenate([np.array([level, hp_frac, *status, *types])] + moves)

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        power = move.base_power / 250
        acc = move.accuracy / 100
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array([power, acc, *move_type])

    @staticmethod
    def get_action_space(battle: AbstractBattle) -> list[int]:
        if battle.active_pokemon is None:
            return []
        else:
            move_space = [
                i
                for i, move in enumerate(battle.active_pokemon.moves.values())
                if move.id in [m.id for m in battle.available_moves]
            ]
            switch_space = [
                i + 4
                for i, pokemon in enumerate(battle.team.values())
                if pokemon.species in [p.species for p in battle.available_switches]
            ]
            return move_space + switch_space
