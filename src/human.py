import json
from typing import Any

import requests


def scrape():
    print("Scraping replay data from pages 1-100")
    battle_jsons = [
        battle_json
        for page in range(1, 101)
        for battle_json in get_battle_jsons(page)
        if battle_json["rating"] is None
    ]
    print(f"Retrieved {len(battle_jsons)} battle header jsons")
    log_jsons = [get_log_json(battle_json["id"][16:]) for battle_json in battle_jsons]
    print(f"Converted to {len(log_jsons)} battle logs")
    logs = [lj["log"] for lj in log_jsons if "|showteam|" in lj["log"]]
    print(f"Saving {len(logs)} battle logs with open sheets to json/human.json")
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
