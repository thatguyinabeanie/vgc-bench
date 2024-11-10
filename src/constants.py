from poke_env.environment import (
    Effect,
    Field,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

from data import abilities, items, moves

singles_act_len = 26
doubles_act_len = 47
pokemon_obs_len = (
    len(abilities)
    + len(items)
    + len(moves)
    + len(Effect)
    + len(PokemonGender)
    + 2 * len(PokemonType)
    + len(Status)
    + 35
)
side_obs_len = 8 * len(moves) + len(SideCondition) + pokemon_obs_len + 4
glob_obs_len = len(Field) + len(Weather) + 2
singles_chunk_len = singles_act_len + glob_obs_len + side_obs_len
doubles_chunk_len = 2 * doubles_act_len + glob_obs_len + side_obs_len
singles_obs_len = 12 * singles_chunk_len
doubles_obs_len = 12 * doubles_chunk_len
