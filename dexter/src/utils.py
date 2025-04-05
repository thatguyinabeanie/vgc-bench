import json
from enum import Enum, auto, unique

import numpy as np
import numpy.typing as npt
from poke_env.environment import (
    Effect,
    Field,
    MoveCategory,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)


@unique
class LearningStyle(Enum):
    EXPLOITER = auto()
    PURE_SELF_PLAY = auto()
    FICTITIOUS_PLAY = auto()
    DOUBLE_ORACLE = auto()

    @property
    def is_self_play(self) -> bool:
        return self in {
            LearningStyle.PURE_SELF_PLAY,
            LearningStyle.FICTITIOUS_PLAY,
            LearningStyle.DOUBLE_ORACLE,
        }

    @property
    def abbrev(self) -> str:
        match self:
            case LearningStyle.EXPLOITER:
                return "ex"
            case LearningStyle.PURE_SELF_PLAY:
                return "sp"
            case LearningStyle.FICTITIOUS_PLAY:
                return "fp"
            case LearningStyle.DOUBLE_ORACLE:
                return "do"


# training params
battle_format = "gen9vgc2025regg"
num_envs = 24
frame_stack = False
num_frames = 3
steps = 98_304

# observation length constants
singles_act_len = 26
doubles_act_len = 107
singles_glob_obs_len = singles_act_len + len(Field) + len(Weather) + 7
doubles_glob_obs_len = 2 * doubles_act_len + len(Field) + len(Weather) + 8
side_obs_len = len(SideCondition) + 5
move_obs_len = len(MoveCategory) + len(Target) + len(PokemonType) + 11
pokemon_obs_len = (
    4 * move_obs_len + len(Effect) + len(PokemonGender) + 2 * len(PokemonType) + len(Status) + 37
)
singles_chunk_obs_len = singles_glob_obs_len + side_obs_len + pokemon_obs_len
doubles_chunk_obs_len = doubles_glob_obs_len + side_obs_len + pokemon_obs_len

# pokemon data
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
