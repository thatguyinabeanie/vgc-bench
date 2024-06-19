import torch
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import BattleOrder, Player

from nn import MLP


class Agent(Player):
    nn: MLP

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.nn = MLP(10, [100, 100], 10)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        embedded_battle = self.embed_battle(battle)
        if isinstance(battle, Battle):
            action_space = self.get_action_space(battle)
            if not action_space or (
                len(battle.available_moves) == 1 and battle.available_moves[0].id == "recharge"
            ):
                return self.choose_default_move()
            assert battle.active_pokemon is not None
            total_action_space = list(battle.active_pokemon.moves.values()) + list(
                battle.team.values()
            )
            output = self.nn(embedded_battle)
            mask = torch.full((10,), float("-inf"))
            mask[action_space] = 0
            soft_output = torch.softmax(output + mask, dim=0)
            choice = int(torch.multinomial(soft_output, 1).item())
            return self.create_order(total_action_space[choice])
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise Exception("Must be single or double battle")

    @staticmethod
    def embed_battle(battle: AbstractBattle) -> torch.Tensor:
        if isinstance(battle, Battle):
            return torch.rand(10)
        elif isinstance(battle, DoubleBattle):
            return torch.rand(10)
        else:
            raise Exception("Must be single or double battle")

    @staticmethod
    def get_action_space(battle: AbstractBattle) -> list[int]:
        if isinstance(battle, Battle):
            assert battle.active_pokemon is not None
            move_action_space = [
                i
                for i, move in enumerate(battle.active_pokemon.moves.values())
                if move.id in [m.id for m in battle.available_moves]
            ]
            switch_action_space = [
                i + 4
                for i, pokemon in enumerate(battle.team.values())
                if pokemon.species in [p.species for p in battle.available_switches]
            ]
            return move_action_space + switch_action_space
        else:
            return []
