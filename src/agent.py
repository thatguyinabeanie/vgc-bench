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
from poke_env.player import BattleOrder, Player
from stable_baselines3.common.policies import BasePolicy


class Agent(Player):
    policy: BasePolicy

    def __init__(self, policy: BasePolicy, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        action_space = self.get_action_space(battle)
        embedded_battle = torch.tensor(self.embed_battle(battle)).view(1, -1).to(self.policy.device)
        if isinstance(battle, Battle):
            if not action_space or (
                len(battle.available_moves) == 1 and battle.available_moves[0].id == "recharge"
            ):
                return self.choose_default_move()
            assert battle.active_pokemon is not None
            _, act_log_probs, _ = self.policy.evaluate_actions(
                embedded_battle, torch.tensor(range(10)).to(self.policy.device)
            )
            act_probs = torch.exp(act_log_probs)
            mask = torch.full((10,), float("-inf")).to(self.policy.device)
            mask[action_space] = 0
            soft_output = torch.softmax(act_probs + mask, dim=0)
            action_id = int(torch.multinomial(soft_output, 1).item())
            if action_id < 4:
                action = list(battle.active_pokemon.moves.values())[action_id]
            else:
                action = list(battle.team.values())[action_id - 4]
            return self.create_order(action)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise TypeError()

    @staticmethod
    def get_action_space(battle: AbstractBattle) -> list[int]:
        if isinstance(battle, Battle):
            assert battle.active_pokemon is not None
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
        else:
            return []

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
