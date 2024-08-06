import json
import os
import re

import requests

with open("json/abilities.json") as f:
    ABILITYDEX = [name for name, details in json.load(f).items() if "isNonstandard" not in details]
with open("json/items.json") as f:
    ITEMDEX = [
        name
        for name, details in json.load(f).items()
        if "isNonstandard" not in details and "isPokeball" not in details
    ]
with open("json/moves.json") as f:
    MOVEDEX = [name for name, details in json.load(f).items() if "isNonstandard" not in details]


def update_json_file(url: str, file: str):
    response = requests.get(f"{url}/{file}")
    if ".json" in file:
        json_text = response.text
    else:
        js_text = response.text
        i = js_text.index("{")
        js_literal = js_text[i:-1]
        json_text = re.sub(r"([{,])([a-zA-Z0-9_]+)(:)", r'\1"\2"\3', js_literal)
        file += "on"
    with open(f"json/{file}", "w") as f:
        f.write(json_text)


if __name__ == "__main__":
    if not os.path.exists("json"):
        os.mkdir("json")
    update_json_file("https://play.pokemonshowdown.com/data", "abilities.js")
    update_json_file("https://play.pokemonshowdown.com/data", "items.js")
    update_json_file("https://play.pokemonshowdown.com/data", "moves.js")
