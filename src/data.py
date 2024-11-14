import json
import os
import re
import warnings

import numpy as np
import numpy.typing as npt
import requests
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def update_desc_embeddings(url: str, file: str, extras: dict[str, dict[str, str]] = {}):
    response = requests.get(f"{url}/{file}")
    if ".json" in file:
        json_text = response.text
    else:
        js_text = response.text
        i = js_text.index("{")
        js_literal = js_text[i:-1]
        json_text = re.sub(r"([{,])([a-zA-Z0-9_]+)(:)", r'\1"\2"\3', js_literal)
        file += "on"
    dex = {k: v for k, v in {**extras, **json.loads(json_text)}.items() if "desc" in v}
    warnings.simplefilter(action="ignore", category=FutureWarning)
    transformer = SentenceTransformer("paraphrase-mpnet-base-v2")
    embeddings = transformer.encode([a["desc"] for a in dex.values()])
    pca = PCA(100)
    reduced_embeddings = pca.fit_transform(embeddings).tolist()  # type: ignore
    with open(f"json/{file}", "w") as f:
        json.dump(dict(zip(dex.keys(), reduced_embeddings)), f)


if __name__ == "__main__":
    if not os.path.exists("json"):
        os.mkdir("json")
    update_desc_embeddings(
        "https://play.pokemonshowdown.com/data", "abilities.js", extras={"null": {"desc": "null"}}
    )
    update_desc_embeddings(
        "https://play.pokemonshowdown.com/data",
        "items.js",
        extras={
            "null": {"desc": "null"},
            "": {"desc": "empty"},
            "unknown_item": {"desc": "unknown item"},
        },
    )
    update_desc_embeddings("https://play.pokemonshowdown.com/data", "moves.js")
with open("json/abilities.json") as f:
    ability_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    abilities = list(ability_descs.keys())
    ability_embeds = list(ability_descs.values())
with open("json/items.json") as f:
    item_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    items = list(item_descs.keys())
    item_embeds = list(item_descs.values())
with open("json/moves.json") as f:
    move_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    moves = list(move_descs.keys())
    move_embeds = list(move_descs.values())
