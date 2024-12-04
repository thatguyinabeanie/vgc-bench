import asyncio
import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
from poke_env import to_id_str
from poke_env.environment import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DoubleBattleOrder, Player, DefaultBattleOrder

from agent import Agent


class HumanPlayer(Player):
    state_action_pairs: list[tuple[npt.NDArray[np.float32], str]]
    next_msg: str | None
    teampreview_draft: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_action_pairs = []
        self.next_msg = None
        self.teampreview_draft = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert self.next_msg is not None
        assert isinstance(battle, DoubleBattle)
        order1 = self.get_order(battle, self.next_msg, False)
        order2 = self.get_order(battle, self.next_msg, True)
        order = DoubleBattleOrder(order1, order2)
        self.state_action_pairs.append((Agent.embed_battle(battle, self.teampreview_draft), str(order)))
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
            move = active.moves[move_name] if move_name in active.moves else active.moves["metronome"]
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
        id1 = self.get_teampreview_order(battle, self.next_msg, True)
        id2 = self.get_teampreview_order(battle, self.next_msg, False)
        all_choices = [str(c) for c in range(1, 7)]
        all_choices.remove(id1)
        all_choices.remove(id2)
        order = f"/team {id1}{id2}{all_choices[0]}{all_choices[1]}"
        self.teampreview_draft = [p.name for i, p in enumerate(battle.team.values()) if i + 1 in [id1, id2]]
        self.state_action_pairs.append((Agent.embed_battle(battle, self.teampreview_draft), order))
        return order

    @staticmethod
    def get_teampreview_order(battle: AbstractBattle, msg: str, is_left: bool) -> str:
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
        return str(index + 1)

    async def follow_log(self, tag: str, log: str, role: str) -> list[tuple[npt.NDArray[np.float32], str]]:
        assert not self.state_action_pairs
        tag = f"battle-{tag}"
        messages = [f">{tag}\n" + m for m in log.split("\n|\n")]
        for i in range(len(messages) - 2):
            self.next_msg = messages[i + 1]
            if i == 0:
                battle = await self._create_battle(f">{tag}".split("-"))
                battle.logger = None
                battle._player_role = role
                battle._teampreview = True
            split_messages = [m.split("|") for m in messages[i].split("\n")]
            await self._handle_battle_message(split_messages)
            if i > 0:
                battle = self.battles[tag]
                self.choose_move(battle)
        pairs = self.state_action_pairs
        self.state_action_pairs = []
        return pairs


def process_logs(
    log_jsons: dict[str, tuple[str, str]], n: int
) -> list[tuple[AbstractBattle, str]]:
    state_action_pairs = []
    for i, (tag, (_, log)) in enumerate(log_jsons.items()):
        if i == n:
            break
        print(f"conversion progress: {round(100 * i / min(n, len(log_jsons)), ndigits=2)}%", end="\r")
        player1 = HumanPlayer(
            battle_format="gen9vgc2024regh", accept_open_team_sheet=True, start_listening=False
        )
        player2 = HumanPlayer(
            battle_format="gen9vgc2024regh", accept_open_team_sheet=True, start_listening=False
        )
        state_action_pairs += asyncio.run(player1.follow_log(tag, log, "p1"))
        state_action_pairs += asyncio.run(player2.follow_log(tag, log, "p2"))
    print("done!")
    return state_action_pairs


def scrape(increment: int):
    if os.path.exists("json/human-logs.json"):
        with open("json/human-logs.json", "r") as f:
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
        if lj is not None
        and lj["log"].count("|poke|p1|") == 6
        and lj["log"].count("|poke|p2|") == 6
        and "|showteam|" in lj["log"]
        and "Zoroark" not in lj["log"]
        and "Zorua" not in lj["log"]
    }
    logs = {**old_logs, **new_logs}
    with open("json/human-logs.json", "w") as f:
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
    format_str = "gen9vgc2024regh"
    response = requests.get(f"{site}/search.json?format={format_str}&before={before}")
    new_battle_jsons = json.loads(response.text)
    before = new_battle_jsons[-1]["uploadtime"] + 1
    battle_idents |= {
        bj["id"]
        for bj in new_battle_jsons
        if bj["id"].startswith(format_str) and bj["rating"] is None
    }
    return battle_idents, before


def get_log_json(ident: str) -> dict[str, Any] | None:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/{ident}.json")
    if response:
        return json.loads(response.text)


if __name__ == "__main__":
    while True:
        scrape(1000)
