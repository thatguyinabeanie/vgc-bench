import asyncio
import json
import os
from typing import Any

import requests
from poke_env import to_id_str
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder, DoubleBattleOrder, Player


class HumanPlayer(Player):
    states: list[AbstractBattle]
    orders: list[BattleOrder | str]
    next_msg: str | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []
        self.orders = []
        self.next_msg = None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert self.next_msg is not None
        order1 = self.get_order(battle, self.next_msg, False)
        order2 = self.get_order(battle, self.next_msg, True)
        order = DoubleBattleOrder(order1, order2)
        self.states.append(battle)
        self.orders.append(order)
        return order

    @staticmethod
    def get_order(battle: AbstractBattle, msg: str, is_right: bool) -> BattleOrder | None:
        pos = "b" if is_right else "a"
        if f"|move|{battle.player_role}{pos}: " in msg:
            active = battle.active_pokemon[int(is_right)]
            assert active is not None, battle.active_pokemon
            protocol_substr = (
                f"|move|{battle.player_role}{pos}: {active.base_species.capitalize()}|"
            )
            start = msg.index(protocol_substr)
            end = msg.index("|", start + len(protocol_substr))
            move = active.moves[to_id_str(msg[start + len(protocol_substr) : end])]
            return BattleOrder(move)
        elif f"|switch|{battle.player_role}{pos}: " in msg:
            start = msg.index(f"|switch|{battle.player_role}{pos}: ")
            end = msg.index("|", start + 13)
            name = msg[start + 13 : end]
            pokemon = [p for p in battle.team.values() if p.base_species.capitalize() == name][0]
            return BattleOrder(pokemon)
        else:
            return None

    def teampreview(self, battle: AbstractBattle) -> str:
        assert self.next_msg is not None
        id1 = self.get_teampreview_order(battle, self.next_msg, True)
        id2 = self.get_teampreview_order(battle, self.next_msg, False)
        all_choices = [str(c) for c in range(1, 7)]
        all_choices.remove(str(id1))
        all_choices.remove(str(id2))
        order = f"/team {id1}{id2}{all_choices[0]}{all_choices[1]}"
        self.states.append(battle)
        self.orders.append(order)
        return order

    @staticmethod
    def get_teampreview_order(battle: AbstractBattle, msg: str, is_left: bool) -> str:
        pos = "a" if is_left else "b"
        start = msg.index(f"|switch|{battle.player_role}{pos}")
        end = msg.index("|", start + 13)
        name = msg[start + 13 : end]
        index = [i for i, p in enumerate(battle.team.values()) if p.name == name][0]
        return str(index + 1)

    async def follow_log(
        self, tag: str, log: str, role: str
    ) -> tuple[list[AbstractBattle], list[BattleOrder | str]]:
        assert not self.states and not self.orders
        tag = f"battle-{tag}"
        messages = [f">{tag}\n" + m for m in log.split("\n|\n")]
        for i in range(len(messages) - 2):
            self.next_msg = messages[i + 1]
            if i == 0:
                battle = await self._create_battle(f">{tag}".split("-"))
                battle._player_role = role
                battle._teampreview = True
            split_messages = [m.split("|") for m in messages[i].split("\n")]
            await self._handle_battle_message(split_messages)
            if i > 0:
                battle = self.battles[tag]
                self.choose_move(battle)
        states = self.states
        orders = self.orders
        self.states = []
        self.orders = []
        return states, orders


def scrape(increment: int):
    if os.path.exists("json/human.json"):
        with open("json/human.json", "r") as f:
            old_logs = json.load(f)
        log_times = [int(time) for time, _ in old_logs.values()]
        before = min(log_times)
        newest = max(log_times)
    else:
        old_logs = {}
        before = 2_000_000_000
        newest = None
    print("total battle logs:", len(old_logs), end="\r")
    battle_idents = get_battle_idents(increment, before + 1, newest)
    log_jsons = [get_log_json(ident) for ident in battle_idents]
    new_logs = {
        lj["id"]: (lj["uploadtime"], lj["log"])
        for lj in log_jsons
        if lj is not None and "|showteam|" in lj["log"]
    }
    logs = {**old_logs, **new_logs}
    with open("json/human.json", "w") as f:
        json.dump(logs, f)


def get_battle_idents(num_battles: int, before: int, newest: int | None) -> list[str]:
    battle_idents = set()
    if newest is not None:
        before_ = 2_000_000_000
        while len(battle_idents) < num_battles and before >= newest:
            battle_idents, before_ = update_battle_idents(battle_idents, before_)
    while len(battle_idents) < num_battles:
        battle_idents, before = update_battle_idents(battle_idents, before)
    return list(battle_idents)[:num_battles]


def update_battle_idents(battle_idents: set[str], before: int) -> tuple[set[str], int]:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/search.json?format=gen9vgc2024regh&before={before}")
    new_battle_jsons = json.loads(response.text)
    before = new_battle_jsons[-1]["uploadtime"] + 1
    battle_idents |= {bj["id"] for bj in new_battle_jsons if bj["rating"] is None}
    return battle_idents, before


def get_log_json(ident: str) -> dict[str, Any] | None:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/{ident}.json")
    if response:
        return json.loads(response.text)


if __name__ == "__main__":
    # while True:
    #     scrape(1000)
    with open("json/human.json", "r") as f:
        log_jsons = json.load(f)
    player1 = HumanPlayer(
        battle_format="gen9vgc2024regh", accept_open_team_sheet=True, start_listening=False
    )
    player2 = HumanPlayer(
        battle_format="gen9vgc2024regh", accept_open_team_sheet=True, start_listening=False
    )
    tag, (time, log) = list(log_jsons.items())[0]
    states1, orders1 = asyncio.run(player1.follow_log(tag, log, "p1"))
    states2, orders2 = asyncio.run(player2.follow_log(tag, log, "p2"))
    print(log)
    print(states1)
    print(states2)
    print(orders1)
    print(orders2)
