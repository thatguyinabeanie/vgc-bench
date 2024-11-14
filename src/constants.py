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
    + 4 * len(moves)
    + len(Effect)
    + len(PokemonGender)
    + 2 * len(PokemonType)
    + len(Status)
    + 35
)
side_obs_len = pokemon_obs_len + len(SideCondition) + 4
glob_obs_len = len(Field) + len(Weather) + 2
chunk_len = side_obs_len + glob_obs_len
singles_obs_len = singles_act_len + 12 * chunk_len
doubles_obs_len = 2 * doubles_act_len + 12 * chunk_len
