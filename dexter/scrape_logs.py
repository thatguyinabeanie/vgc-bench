import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

battle_formats = [
    "gen9vgc2023regd",
    "gen9vgc2024regf",
    "gen9vgc2024regfbo3",
    "gen9vgc2024regg",
    "gen9vgc2024reggbo3",
    "gen9vgc2024regh",
    "gen9vgc2024reghbo3",
    "gen9vgc2025regg",
    "gen9vgc2025reggbo3",
]


def scrape_logs(increment: int, battle_format: str) -> bool:
    if os.path.exists(f"data/logs-{battle_format}.json"):
        with open(f"data/logs-{battle_format}.json", "r") as f:
            old_logs = json.load(f)
    else:
        old_logs = {}
    log_times = [int(time) for f, (time, _) in old_logs.items() if f.startswith(battle_format)]
    oldest = min(log_times) if log_times else 2_000_000_000
    newest = max(log_times) if log_times else None
    battle_idents = get_battle_idents(increment, battle_format, oldest, newest)
    battle_idents = [ident for ident in battle_idents if ident not in old_logs.keys()]
    with ThreadPoolExecutor(max_workers=8) as executor:
        log_jsons = list(executor.map(get_log_json, battle_idents))
    new_logs = {
        lj["id"]: (lj["uploadtime"], lj["log"])
        for lj in log_jsons
        if lj is not None
        and lj["log"].count("|poke|p1|") == 6
        and lj["log"].count("|poke|p2|") == 6
        and "|showteam|" in lj["log"]
        and "Zoroark" not in lj["log"]
        and "Zorua" not in lj["log"]
        and "|-mega|" not in lj["log"]
    }
    logs = {**old_logs, **new_logs}
    print(f"{battle_format}:", len(logs))
    with open(f"data/logs-{battle_format}.json", "w") as f:
        json.dump(logs, f)
    return len(logs) == len(old_logs)


def get_battle_idents(
    num_battles: int, battle_format: str, oldest: int, newest: int | None
) -> set[str]:
    battle_idents = set()
    # Collecting games that happened after we first started collecting
    if newest is not None:
        oldest_ = 2_000_000_000
        while oldest_ >= newest:
            battle_idents, oldest_ = update_battle_idents(battle_idents, battle_format, oldest_)
    # Collecting games that are older than anything we've seen yet
    while len(battle_idents) < num_battles:
        o = oldest
        battle_idents, oldest = update_battle_idents(battle_idents, battle_format, oldest)
        if oldest == o:
            break
    return battle_idents


def update_battle_idents(
    battle_idents: set[str], battle_format: str, oldest: int
) -> tuple[set[str], int]:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/search.json?format={battle_format}&before={oldest + 1}")
    new_battle_jsons = json.loads(response.text)
    oldest = new_battle_jsons[-1]["uploadtime"]
    battle_idents |= {bj["id"] for bj in new_battle_jsons if bj["id"].startswith(battle_format)}
    return battle_idents, oldest


def get_log_json(ident: str) -> dict[str, Any] | None:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/{ident}.json")
    if response:
        return json.loads(response.text)


if __name__ == "__main__":

    def run(f: str):
        done = False
        while not done:
            done = scrape_logs(4000, f)
        print(f"{f} finished")

    with ThreadPoolExecutor(max_workers=len(battle_formats)) as executor:
        executor.map(run, battle_formats)
