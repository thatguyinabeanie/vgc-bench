from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import BattleOrder, Player


class BasicPlayer(Player):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, Battle):
            return self.choose_random_singles_move(battle)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise Exception("Must be single or double battle")
