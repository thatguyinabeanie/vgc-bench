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
            return self.action_to_move(int(action.item()), battle)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise TypeError()

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        # copied directly from Gen4EnvSinglePlayer class from poke-env
        if action == -1:
            return ForfeitBattleOrder()
        elif isinstance(battle, Battle):
            if action not in battle.action_space:
                return self.choose_random_move(battle)
            elif action < 4:
                assert battle.active_pokemon is not None
                return self.create_order(list(battle.active_pokemon.moves.values())[action])
            else:
                return self.create_order(list(battle.team.values())[action - 4])
        else:
            return self.choose_random_move(battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
        if isinstance(battle, Battle):
            return np.concatenate(
                [Agent.embed_pokemon(p) for p in battle.team.values()]
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
