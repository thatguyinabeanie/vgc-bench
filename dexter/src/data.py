import json

import numpy as np
import numpy.typing as npt

with open("data/abilities.json") as f:
    ability_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    abilities = list(ability_descs.keys())
    ability_embeds = list(ability_descs.values())
with open("data/items.json") as f:
    item_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    items = list(item_descs.keys())
    item_embeds = list(item_descs.values())
with open("data/moves.json") as f:
    move_descs: dict[str, npt.NDArray[np.float32]] = json.load(f)
    moves = list(move_descs.keys())
    move_embeds = list(move_descs.values())
