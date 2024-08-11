import os
import re

import requests

ABILITYDEX = [
    "?",
    None,
    "contrary",
    "protosynthesis",
    "supremeoverlord",
    "magicbounce",
    "toxicdebris",
    "intimidate",
]
ITEMDEX = ["?", None, "", "unknown_item", "leftovers", "boosterenergy", "focussash", "rockyhelmet"]
MOVEDEX = [
    "?",
    "substitute",
    "leafstorm",
    "terablast",
    "glare",
    "dragondance",
    "outrage",
    "flareblitz",
    "morningsun",
    "swordsdance",
    "lowkick",
    "suckerpunch",
    "ironhead",
    "calmmind",
    "drainingkiss",
    "psyshock",
    "mysticalfire",
    "spikes",
    "mortalspin",
    "earthpower",
    "powergem",
    "stealthrock",
    "earthquake",
    "uturn",
    "grassknot",
]
POKEDEX = ["?", "serperior", "gougingfire", "kingambit", "hatterene", "glimmora", "landorustherian"]


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
    update_json_file("https://play.pokemonshowdown.com/data", "pokedex.js")
