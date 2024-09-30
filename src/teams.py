# teams from https://www.reddit.com/r/VGC/comments/1f7b9dy/regulation_h_rental_teams/

import random
from subprocess import run

from poke_env.teambuilder import Teambuilder


class RandomTeamBuilder(Teambuilder):
    teams: list[str]

    def __init__(self, num_teams: int, battle_format: str):
        self.teams = []
        for team in TEAMS[:num_teams]:
            result = run(
                ["node", "pokemon-showdown", "validate-team", battle_format],
                input=f'"{team[1:]}"'.encode(),
                cwd="pokemon-showdown",
                capture_output=True,
            )
            if result.returncode == 1:
                print(f"team {TEAMS.index(team)}: {result.stderr.decode()}")
            else:
                parsed_team = self.parse_showdown_team(team)
                packed_team = self.join_team(parsed_team)
                self.teams.append(packed_team)

    def yield_team(self) -> str:
        return random.choice(self.teams)


TEAMS = [
    """
Typhlosion-Hisui @ Charcoal
Ability: Blaze
Level: 50
Tera Type: Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Eruption
- Heat Wave
- Shadow Ball
- Protect

Whimsicott @ Covert Cloak
Ability: Prankster
Level: 50
Tera Type: Dark
EVs: 244 HP / 4 Def / 4 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Tailwind
- Moonblast
- Sunny Day
- Encore

Ursaluna-Bloodmoon @ Life Orb
Ability: Mind's Eye
Level: 50
Tera Type: Normal
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Hyper Voice
- Blood Moon
- Earth Power
- Protect

Primarina @ Throat Spray
Ability: Liquid Voice
Level: 50
Tera Type: Grass
EVs: 244 HP / 52 Def / 108 SpA / 28 SpD / 76 Spe
Modest Nature
IVs: 0 Atk
- Moonblast
- Hyper Voice
- Haze
- Protect

Meowscarada @ Focus Sash
Ability: Protean
Level: 50
Tera Type: Grass
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Flower Trick
- Knock Off
- U-turn
- Protect

Farigiraf @ Safety Goggles
Ability: Armor Tail
Level: 50
Tera Type: Fire
EVs: 228 HP / 156 Def / 124 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Psychic Noise
- Hyper Voice
- Helping Hand
- Trick Room
""",
    """
Tyranitar @ Assault Vest
Ability: Sand Stream
Level: 50
Tera Type: Flying
EVs: 124 HP / 252 Atk / 36 Def / 12 SpD / 84 Spe
Adamant Nature
- Rock Slide
- Knock Off
- Tera Blast
- Low Kick

Excadrill @ Clear Amulet
Ability: Sand Rush
Level: 50
Shiny: Yes
Tera Type: Ghost
EVs: 180 HP / 116 Atk / 20 Def / 28 SpD / 164 Spe
Adamant Nature
- Earthquake
- Iron Head
- High Horsepower
- Protect

Sinistcha @ Rocky Helmet
Ability: Hospitality
Level: 50
Tera Type: Dark
EVs: 236 HP / 36 Def / 236 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Matcha Gotcha
- Life Dew
- Trick Room
- Rage Powder

Dragonite @ Choice Band
Ability: Inner Focus
Level: 50
Tera Type: Normal
EVs: 68 HP / 252 Atk / 4 Def / 4 SpD / 180 Spe
Adamant Nature
- Outrage
- Extreme Speed
- Ice Spinner
- Aerial Ace

Primarina @ Throat Spray
Ability: Liquid Voice
Level: 50
Tera Type: Grass
EVs: 252 HP / 68 Def / 108 SpA / 4 SpD / 76 Spe
Modest Nature
IVs: 0 Atk
- Moonblast
- Hyper Voice
- Haze
- Protect

Volcarona @ Leftovers
Ability: Flame Body
Level: 50
Tera Type: Dragon
EVs: 252 HP / 28 Def / 36 SpA / 4 SpD / 188 Spe
Modest Nature
IVs: 0 Atk
- Fiery Dance
- Giga Drain
- Quiver Dance
- Protect""",
    """
Annihilape @ Lum Berry
Ability: Defiant
Level: 50
Tera Type: Water
EVs: 180 HP / 36 Atk / 12 Def / 28 SpD / 252 Spe
Adamant Nature
- Rage Fist
- Drain Punch
- Bulk Up
- Protect

Maushold @ Safety Goggles
Ability: Friend Guard
Level: 50
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Follow Me
- Beat Up
- Taunt
- Protect

Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
Tera Type: Fairy
EVs: 236 HP / 36 Def / 236 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Matcha Gotcha
- Life Dew
- Trick Room
- Rage Powder

Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Grass
EVs: 212 HP / 12 Def / 44 SpA / 212 SpD / 28 Spe
Modest Nature
- Electro Shot
- Flash Cannon
- Body Press
- Draco Meteor

Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Stellar
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Hurricane
- Weather Ball
- Wide Guard
- Protect

Hydreigon @ Scope Lens
Ability: Levitate
Level: 50
Shiny: Yes
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Focus Energy
- Protect""",
    """
Indeedee-F @ Psychic Seed
Ability: Psychic Surge
Level: 50
Tera Type: Grass
EVs: 244 HP / 244 Def / 20 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Follow Me
- Psychic
- Helping Hand
- Trick Room

Hatterene @ Life Orb
Ability: Magic Bounce
Level: 50
Tera Type: Fire
EVs: 212 HP / 44 Def / 252 SpA
Quiet Nature
IVs: 0 Atk / 0 Spe
- Expanding Force
- Dazzling Gleam
- Trick Room
- Protect

Torkoal @ Charcoal
Ability: Drought
Level: 50
Tera Type: Fire
EVs: 244 HP / 252 SpA / 12 SpD
Quiet Nature
IVs: 0 Atk / 0 Spe
- Eruption
- Weather Ball
- Clear Smog
- Protect

Lilligant-Hisui @ Focus Sash
Ability: Chlorophyll
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Sleep Powder
- After You
- Leaf Blade
- Close Combat

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Ghost
EVs: 140 HP / 236 Atk / 132 SpD
Brave Nature
IVs: 0 Spe
- Headlong Rush
- Facade
- Swords Dance
- Protect

Gallade @ Clear Amulet
Ability: Sharpness
Level: 50
Tera Type: Grass
EVs: 252 HP / 196 Atk / 60 Def
Brave Nature
IVs: 0 Spe
- Sacred Sword
- Psycho Cut
- Wide Guard
- Trick Room""",
    """
Dragonite @ Choice Band
Ability: Inner Focus
Level: 50
Tera Type: Normal
EVs: 196 HP / 252 Atk / 4 Def / 4 SpD / 52 Spe
Adamant Nature
- Extreme Speed
- Ice Spinner
- Aerial Ace
- Outrage

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Thunderbolt
- Trick

Talonflame @ Covert Cloak
Ability: Gale Wings
Level: 50
Tera Type: Flying
EVs: 4 HP / 244 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Tailwind
- Taunt
- Brave Bird
- Will-O-Wisp

Glimmora @ Power Herb
Ability: Toxic Debris
Level: 50
Tera Type: Grass
EVs: 12 HP / 4 Def / 236 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Sludge Bomb
- Meteor Beam
- Earth Power
- Spiky Shield

Dondozo @ Leftovers
Ability: Unaware
Level: 50
Tera Type: Grass
EVs: 20 HP / 252 Atk / 4 Def / 4 SpD / 228 Spe
Jolly Nature
- Order Up
- Wave Crash
- Tera Blast
- Protect

Tatsugiri @ Choice Scarf
Ability: Commander
Level: 50
Tera Type: Water
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Muddy Water
- Icy Wind
- Dragon Pulse""",
    """
Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Stellar
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Hurricane
- Weather Ball
- Wide Guard
- Protect

Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Grass
EVs: 212 HP / 20 Def / 36 SpA / 212 SpD / 28 Spe
Modest Nature
- Draco Meteor
- Electro Shot
- Flash Cannon
- Body Press

Amoonguss @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Fairy
EVs: 244 HP / 196 Def / 68 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Rage Powder
- Pollen Puff
- Spore
- Clear Smog

Basculegion @ Choice Band
Ability: Swift Swim
Level: 50
Tera Type: Grass
EVs: 100 HP / 252 Atk / 12 Def / 12 SpD / 132 Spe
Adamant Nature
- Wave Crash
- Last Respects
- Tera Blast
- Flip Turn

Salamence @ Life Orb
Ability: Intimidate
Level: 50
Tera Type: Flying
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Hurricane
- Tailwind
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Tera Type: Dark
EVs: 212 HP / 252 Atk / 4 Def / 12 SpD / 28 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Swords Dance
- Protect""",
    """
Baxcalibur @ Clear Amulet
Ability: Thermal Exchange
Level: 50
Tera Type: Water
EVs: 140 HP / 196 Atk / 12 Def / 60 SpD / 100 Spe
Adamant Nature
- Icicle Crash
- Glaive Rush
- Ice Shard
- Protect

Ninetales-Alola @ Light Clay
Ability: Snow Warning
Level: 50
Tera Type: Ghost
EVs: 220 HP / 20 Def / 4 SpA / 12 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Blizzard
- Icy Wind
- Aurora Veil
- Moonblast

Gholdengo @ Leftovers
Ability: Good as Gold
Level: 50
Tera Type: Dragon
EVs: 236 HP / 4 Def / 52 SpA / 12 SpD / 204 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Nasty Plot
- Protect

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 132 HP / 196 Atk / 28 Def / 76 SpD / 76 Spe
Adamant Nature
- Fake Out
- Wood Hammer
- Grassy Glide
- U-turn

Volcarona @ Sitrus Berry
Ability: Flame Body
Level: 50
Tera Type: Grass
EVs: 252 HP / 60 Def / 36 SpA / 4 SpD / 156 Spe
Modest Nature
IVs: 0 Atk
- Heat Wave
- Giga Drain
- Quiver Dance
- Protect

Tauros-Paldea-Aqua @ Mirror Herb
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Raging Bull
- Close Combat
- Aqua Jet
- Protect""",
    """
Kommo-o @ Throat Spray
Ability: Overcoat
Level: 50
Tera Type: Steel
EVs: 44 HP / 4 Def / 252 SpA / 4 SpD / 204 Spe
Timid Nature
IVs: 0 Atk
- Clanging Scales
- Clangorous Soul
- Flash Cannon
- Protect

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Ghost
EVs: 236 HP / 4 Atk / 36 Def / 196 SpD / 36 Spe
Careful Nature
- Fake Out
- Knock Off
- Parting Shot
- Taunt

Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
Tera Type: Fairy
EVs: 236 HP / 36 Def / 236 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Matcha Gotcha
- Life Dew
- Trick Room
- Rage Powder

Primarina @ Mystic Water
Ability: Liquid Voice
Level: 50
Tera Type: Grass
EVs: 244 HP / 52 Def / 196 SpA / 4 SpD / 12 Spe
Modest Nature
IVs: 0 Atk
- Hyper Voice
- Moonblast
- Haze
- Protect

Porygon2 @ Eviolite
Ability: Download
Level: 50
Tera Type: Flying
EVs: 252 HP / 4 Atk / 124 Def / 92 SpA / 36 SpD
Quiet Nature
- Tera Blast
- Ice Beam
- Recover
- Trick Room

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Ghost
EVs: 140 HP / 236 Atk / 132 SpD
Brave Nature
IVs: 0 Spe
- Headlong Rush
- Facade
- Substitute
- Protect""",
]
