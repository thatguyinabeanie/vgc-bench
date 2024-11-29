import json
from typing import Any

import requests


def scrape():
    battle_jsons = [
        battle_json
        for page in range(1, 101)
        for battle_json in get_battle_jsons(page)
        if battle_json["rating"] is None
    ]
    log_jsons = [get_log_json(battle_json["id"][16:]) for battle_json in battle_jsons]
    logs = [lj["log"] for lj in log_jsons if "|showteam|" in lj["log"]]
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
