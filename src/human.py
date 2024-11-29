import json
import os
from typing import Any

import requests


def scrape():
    if os.path.exists("json/human.json"):
        with open("json/human.json", "r") as f:
            old_logs = json.load(f)
        print(f"Loaded {len(old_logs)} logs from json/human.json")
    else:
        old_logs = {}
    battle_jsons = [bj for p in range(1, 101) for bj in get_battle_jsons(p) if bj["rating"] is None]
    print(f"Scraped {len(battle_jsons)} battle header jsons")
    log_jsons = [get_log_json(bj["id"][16:]) for bj in battle_jsons]
    new_logs = {lj["id"]: lj["log"] for lj in log_jsons if "|showteam|" in lj["log"]}
    print(f"Filtered down to {len(new_logs)} battle logs with open sheets")
    logs = {**old_logs, **new_logs}
    print(f"Total new logs gathered: {len(logs) - len(old_logs)}")
    with open("json/human.json", "w") as f:
        json.dump(logs, f)


def get_battle_jsons(page: int) -> Any:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/search.json?format=gen9vgc2024regh&page={page}")
    return json.loads(response.text)


def get_log_json(ident: str) -> Any:
    site = "https://replay.pokemonshowdown.com"
    response = requests.get(f"{site}/gen9vgc2024regh-{ident}.json")
    return json.loads(response.text)


if __name__ == "__main__":
    scrape()
