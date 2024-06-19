import torch
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import BattleOrder, Player

from experience import Experience
from nn import MLP


class Agent(Player):
    nn: MLP
    experiences: list[tuple[torch.Tensor, int, float, bool]]

    def __init__(self, nn: MLP, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.nn = nn
        self.experiences = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        embedded_battle = self.embed_battle(battle)
        reward = 1 if battle.won else -1 if battle.lost else 0
        if isinstance(battle, Battle):
            action_space = self.get_action_space(battle)
            if not action_space or (
                len(battle.available_moves) == 1 and battle.available_moves[0].id == "recharge"
            ):
                self.experiences += [(embedded_battle, 0, reward, battle.finished)]
                return self.choose_default_move()
            assert battle.active_pokemon is not None
            output = self.nn(embedded_battle)
            mask = torch.full((10,), float("-inf")).to(self.nn.device)
            mask[action_space] = 0
            soft_output = torch.softmax(output + mask, dim=0)
            action_id = int(torch.multinomial(soft_output, 1).item())
            if action_id < 4:
                action = list(battle.active_pokemon.moves.values())[action_id]
            else:
                action = list(battle.team.values())[action_id - 4]
            self.experiences += [(embedded_battle, action_id, reward, battle.finished)]
            return self.create_order(action)
        elif isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        else:
            raise Exception("Must be single or double battle")

    def embed_battle(self, battle: AbstractBattle) -> torch.Tensor:
        if isinstance(battle, Battle):
            return torch.rand(10).to(self.nn.device)
        elif isinstance(battle, DoubleBattle):
            return torch.rand(10).to(self.nn.device)
        else:
            raise Exception("Must be single or double battle")

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

    def process_experiences(self) -> list[Experience]:
        experiences: list[Experience] = []
        for i in range(len(self.experiences) - 2):
            state, action, _, done1 = self.experiences[i]
            next_state, _, reward, done2 = self.experiences[i + 1]
            if not done1:
                experiences += [Experience(state, action, next_state, reward, done2)]
        return experiences
