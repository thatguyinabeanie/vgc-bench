from poke_env.environment import (
    Effect,
    Field,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

singles_act_len = 26
doubles_act_len = 47
singles_glob_obs_len = singles_act_len + len(Field) + len(Weather) + 7
doubles_glob_obs_len = 2 * doubles_act_len + len(Field) + len(Weather) + 8
side_obs_len = len(SideCondition) + 4
pokemon_obs_len = len(Effect) + len(PokemonGender) + 2 * len(PokemonType) + len(Status) + 41
singles_chunk_obs_len = singles_glob_obs_len + side_obs_len + pokemon_obs_len
doubles_chunk_obs_len = doubles_glob_obs_len + side_obs_len + pokemon_obs_len
