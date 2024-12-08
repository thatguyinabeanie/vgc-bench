import numpy as np
import numpy.typing as npt
from poke_env import to_id_str
from poke_env.environment import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DefaultBattleOrder, DoubleBattleOrder, Player
from src.agent import Agent
from src.constants import doubles_act_len


class LogReader(Player):
    obs: list[npt.NDArray[np.float32]]
    logits: list[npt.NDArray[np.float32]]
    next_msg: str | None
    teampreview_draft: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(start_listening=False, *args, **kwargs)
        self.obs = []
        self.logits = []
        self.next_msg = None
        self.teampreview_draft = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert self.next_msg is not None
        assert isinstance(battle, DoubleBattle)
        order1 = self.get_order(battle, self.next_msg, False)
        order2 = self.get_order(battle, self.next_msg, True)
        order = DoubleBattleOrder(order1, order2)
        self.embed_data_pair(battle, order)
        return order

    @staticmethod
    def get_order(battle: DoubleBattle, msg: str, is_right: bool) -> BattleOrder | None:
        pos = "b" if is_right else "a"
        move_lines = [
            l
            for l in msg.split("\n")
            if f"|move|{battle.player_role}{pos}: " in l and "[from]" not in l
        ]
        if move_lines:
            active = battle.active_pokemon[int(is_right)]
            assert active is not None, battle.active_pokemon
            move_msg_parts = move_lines[0].split("|")
            if to_id_str(move_msg_parts[3]) == "struggle":
                return DefaultBattleOrder()
            move_name = to_id_str(move_msg_parts[3])
            move = (
                active.moves[move_name] if move_name in active.moves else active.moves["metronome"]
            )
            labels = ["p1a", "p1b", "p2a", "p2b"]
            target = (
                None
                if move_msg_parts
                else [
                    p
                    for i, p in enumerate(battle.active_pokemon + battle.opponent_active_pokemon)
                    if p is not None
                    and f"{labels[i]}: {p.base_species.capitalize()}" == move_msg_parts[4]
                ][0]
            )
            did_tera = f"|-terastallize|{move_msg_parts[2]}|" in msg
            return BattleOrder(
                move, terastallize=did_tera, move_target=battle.to_showdown_target(move, target)
            )
        elif f"|switch|{battle.player_role}{pos}: " in msg:
            start = msg.index(f"|switch|{battle.player_role}{pos}: ")
            end = msg.index("\n", start) if "\n" in msg[start:] else len(msg)
            switch_msg_parts = msg[start:end].split("|")
            form = switch_msg_parts[3].split(", ")[0]
            pokemon = [
                p
                for p in battle.team.values()
                if to_id_str(form).startswith(to_id_str(p._last_details.split(", ")[0]))
                or to_id_str(p._last_details.split(", ")[0]).startswith(to_id_str(form))
            ][0]
            return BattleOrder(pokemon)
        else:
            return None

    def teampreview(self, battle: AbstractBattle) -> str:
        assert self.next_msg is not None
        assert isinstance(battle, DoubleBattle)
        id1 = self.get_teampreview_order(battle, self.next_msg, True)
        id2 = self.get_teampreview_order(battle, self.next_msg, False)
        all_choices = [str(c) for c in range(1, 7)]
        all_choices.remove(str(id1))
        all_choices.remove(str(id2))
        order_str = f"/team {id1}{id2}{all_choices[0]}{all_choices[1]}"
        order1 = BattleOrder(list(battle.team.values())[id1 - 1])
        order2 = BattleOrder(list(battle.team.values())[id2 - 1])
        order = DoubleBattleOrder(order1, order2)
        self.embed_data_pair(battle, order)
        self.teampreview_draft = [
            p.name for i, p in enumerate(battle.team.values()) if i + 1 in [id1, id2]
        ]
        return order_str

    @staticmethod
    def get_teampreview_order(battle: AbstractBattle, msg: str, is_left: bool) -> int:
        pos = "a" if is_left else "b"
        start = msg.index(f"|switch|{battle.player_role}{pos}: ")
        end = msg.index("\n", start)
        switch_msg_parts = msg[start:end].split("|")
        form = switch_msg_parts[3].split(", ")[0]
        index = [
            i
            for i, p in enumerate(battle.team.values())
            if to_id_str(form).startswith(to_id_str(p._last_details.split(", ")[0]))
        ][0]
        return index + 1

    def embed_data_pair(self, battle: DoubleBattle, order: DoubleBattleOrder):
        obs = Agent.embed_battle(battle, self.teampreview_draft)
        action1, action2 = Agent.doubles_order_to_action(order, battle)
        action_logits = np.zeros(2 * doubles_act_len, dtype=np.float32)
        action_logits[action1] = 1
        action_logits[action2] = 1
        self.obs += [obs]
        self.logits += [action_logits]

    async def follow_log(
        self, tag: str, log: str, role: str
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        self.obs = []
        self.logits = []
        self.teampreview_draft = []
        tag = f"battle-{tag}"
        messages = [f">{tag}\n" + m for m in log.split("\n|\n")]
        for i in range(len(messages) - 1):
            split_messages = [m.split("|") for m in messages[i].split("\n")]
            self.next_msg = messages[i + 1]
            if i == 0:
                battle = await self._create_battle(f">{tag}".split("-"))
                battle.logger = None
                battle._player_role = role
                battle._teampreview = True
                await self._handle_battle_message(split_messages)
            else:
                battle = self.battles[tag]
                battle._teampreview = False
                await self._handle_battle_message(split_messages)
                self.choose_move(battle)
        split_messages = [m.split("|") for m in messages[-1].split("\n")]
        await self._handle_battle_message(split_messages)
        last_obs = Agent.embed_battle(self.battles[tag], self.teampreview_draft)
        self.obs += [last_obs]
        return np.stack(self.obs, axis=0), np.stack(self.logits, axis=0)
