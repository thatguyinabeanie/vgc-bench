import torch
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import BattleOrder, Player

from experience import Experience
from nn import MLP


class Agent(Player):
    nn: MLP
    battle_records: dict[str, list[tuple[torch.Tensor, int]]]
    experiences: list[Experience]

    def __init__(self, nn: MLP, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.nn = nn
        self.battle_records = {}
        self.experiences = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.turn == 1:
            self.battle_records[battle.battle_tag] = []
        action_space = self.get_action_space(battle)
        embedded_battle = self.embed_battle(battle).to(self.nn.device)
        if isinstance(battle, Battle):
            if not action_space or (
                len(battle.available_moves) == 1 and battle.available_moves[0].id == "recharge"
            ):
                self.battle_records[battle.battle_tag] += [(embedded_battle, -1)]
                return self.choose_default_move()
            assert battle.active_pokemon is not None
            output = self.nn(embedded_battle)
            mask = torch.full((10,), float("-inf")).to(self.nn.device)
            mask[action_space] = 0
            soft_output = torch.softmax(output + mask, dim=0)
            action_id = int(torch.multinomial(soft_output, 1).item())
            self.battle_records[battle.battle_tag] += [(embedded_battle, action_id)]
            if action_id < 4:
                action = list(battle.active_pokemon.moves.values())[action_id]
            else:
                action = list(battle.team.values())[action_id - 4]
            return self.create_order(action)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise Exception("Must be single or double battle")

    def reset_battles(self):
        for tag, records in self.battle_records.items():
            for i in range(len(records) - 1):
                state, action = records[i]
                next_state, _ = records[i + 1]
                if action != -1:
                    done = i == len(records) - 2
                    reward = (
                        1
                        if self.battles[tag].won and done
                        else -1 if self.battles[tag].lost and done else 0
                    )
                    self.experiences += [Experience(state, action, next_state, reward, done)]
        self.battle_records = {}
        super().reset_battles()

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
    def embed_battle(battle: AbstractBattle) -> torch.Tensor:
        if isinstance(battle, Battle):
            return torch.rand(10)
        elif isinstance(battle, DoubleBattle):
            return torch.rand(10)
        else:
            raise Exception("Must be single or double battle")
