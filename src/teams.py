import random
from subprocess import run

from poke_env.teambuilder import Teambuilder


class RandomTeamBuilder(Teambuilder):
    teams: list[str]

    def __init__(self, battle_format: str):
        self.teams = []
        for team in TEAMS:
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
Tommy (Serperior) @ Leftovers
Ability: Contrary
Shiny: Yes
Tera Type: Ground
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Substitute
- Leaf Storm
- Tera Blast
- Glare

Arthur (Gouging Fire) @ Booster Energy
Ability: Protosynthesis
Tera Type: Fairy
EVs: 168 Atk / 88 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Outrage
- Flare Blitz
- Morning Sun

John (Kingambit) @ Leftovers
Ability: Supreme Overlord
Tera Type: Fighting
EVs: 160 HP / 252 Atk / 96 Spe
Adamant Nature
- Swords Dance
- Low Kick
- Sucker Punch
- Iron Head

Polly (Hatterene) (F) @ Leftovers
Ability: Magic Bounce
Tera Type: Water
EVs: 252 HP / 200 Def / 56 Spe
Bold Nature
IVs: 0 Atk
- Calm Mind
- Draining Kiss
- Psyshock
- Mystical Fire

Johnny Dogs (Glimmora) @ Focus Sash
Ability: Toxic Debris
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Spikes
- Mortal Spin
- Earth Power
- Power Gem

Charlie (Landorus-Therian) (M) @ Rocky Helmet
Ability: Intimidate
Tera Type: Water
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Stealth Rock
- Earthquake
- U-turn
- Grass Knot""",
    """
Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Fighting
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Swords Dance
- Iron Head
- Low Kick
- Sucker Punch

Kyurem @ Choice Specs
Ability: Pressure
Tera Type: Fairy
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Ice Beam
- Freeze-Dry
- Earth Power

Scizor @ Choice Band
Ability: Technician
Tera Type: Bug
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Bullet Punch
- Close Combat
- Knock Off
- U-turn

Rotom-Wash @ Leftovers
Ability: Levitate
Tera Type: Steel
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
IVs: 0 Atk
- Volt Switch
- Hydro Pump
- Will-O-Wisp
- Pain Split

Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Poison
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Spikes
- Ruination
- Earthquake
- Whirlwind

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Ground
EVs: 252 Atk / 4 SpD / 252 Spe
- Headlong Rush
- Ice Spinner
- Knock Off
- Rapid Spin""",
    """
Gouging Fire @ Grassy Seed
Ability: Protosynthesis
Tera Type: Poison
EVs: 248 HP / 108 SpD / 152 Spe
Careful Nature
- Flare Blitz
- Breaking Swipe
- Morning Sun
- Dragon Dance

Gholdengo @ Grassy Seed
Ability: Good as Gold
Tera Type: Fairy
EVs: 252 HP / 44 Def / 8 SpA / 100 SpD / 104 Spe
Modest Nature
IVs: 0 Atk
- Nasty Plot
- Shadow Ball
- Make It Rain
- Recover

Rillaboom (F) @ Assault Vest
Ability: Grassy Surge
Tera Type: Fire
EVs: 248 HP / 176 Atk / 32 SpD / 52 Spe
Adamant Nature
- Grassy Glide
- Knock Off
- Low Kick
- U-turn

Zamazenta @ Mirror Herb
Ability: Dauntless Shield
Tera Type: Steel
EVs: 40 HP / 120 Atk / 180 Def / 168 Spe
Jolly Nature
- Body Press
- Heavy Slam
- Crunch
- Iron Defense

Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Steel
EVs: 240 HP / 36 Def / 232 Spe
Timid Nature
- Earth Power
- Stealth Rock
- Taunt
- U-turn

Dragapult @ Focus Sash
Ability: Infiltrator
Tera Type: Fairy
EVs: 76 Atk / 180 SpA / 252 Spe
Naive Nature
IVs: 18 HP
- Dragon Darts
- Hex
- Will-O-Wisp
- Thunder Wave""",
    """
Toxapex @ Leftovers
Ability: Merciless
Tera Type: Ghost
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 0 Atk
- Toxic
- Hex
- Venoshock
- Surf

Glimmora @ Focus Sash
Ability: Toxic Debris
Tera Type: Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Mortal Spin
- Earth Power
- Sludge Wave
- Power Gem

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 252 HP / 196 Def / 60 Spe
Bold Nature
IVs: 0 Atk
- Nasty Plot
- Hex
- Recover
- Make It Rain

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 248 Def / 16 SpD
Impish Nature
- Earthquake
- Spikes
- Protect
- Toxic

Iron Moth @ Booster Energy
Ability: Quark Drive
Tera Type: Fairy
EVs: 124 Def / 132 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Fiery Dance
- Venoshock
- Dazzling Gleam
- Energy Ball

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Ice
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Bulk Up
- Headlong Rush
- Ice Spinner
- Close Combat""",
    """
Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Fairy
EVs: 252 HP / 52 Def / 204 SpD
Careful Nature
- Curse
- Salt Cure
- Recover
- Earthquake

Corviknight @ Rocky Helmet
Ability: Pressure
Tera Type: Dragon
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 27 Spe
- Brave Bird
- Defog
- U-turn
- Roost

Clodsire @ Heavy-Duty Boots
Ability: Water Absorb
Tera Type: Steel
EVs: 252 HP / 252 SpD / 4 Spe
Careful Nature
- Earthquake
- Stealth Rock
- Poison Jab
- Recover

Ditto @ Choice Scarf
Ability: Imposter
Tera Type: Stellar
EVs: 252 HP / 252 Def
Impish Nature
IVs: 30 Atk
- Transform

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Kowtow Cleave
- Iron Head
- Low Kick
- Sucker Punch

SlodogChillionaire (Slowking-Galar) (M) @ Black Sludge
Ability: Regenerator
Shiny: Yes
Tera Type: Water
EVs: 248 HP / 8 Def / 252 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Trick
- Future Sight
- Sludge Bomb
- Chilly Reception""",
    """
Iron Boulder @ Booster Energy
Ability: Quark Drive
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Mighty Cleave
- Zen Headbutt
- Close Combat

Samurott-Hisui @ Focus Sash
Ability: Sharpness
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ceaseless Edge
- Knock Off
- Aqua Cutter
- Aqua Jet

Ogerpon-Wellspring (F) @ Wellspring Mask
Ability: Water Absorb
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Ivy Cudgel
- Power Whip
- Knock Off

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Rapid Spin
- Headlong Rush
- Ice Spinner
- Knock Off

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Fairy
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Hex
- Will-O-Wisp
- U-turn

Kingambit @ Air Balloon
Ability: Supreme Overlord
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Sucker Punch
- Kowtow Cleave
- Iron Head""",
    """
Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Flying
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
IVs: 0 Atk
- Thunder
- Hurricane
- Hydro Pump
- Agility

Pelipper @ Damp Rock
Ability: Drizzle
Tera Type: Ground
EVs: 248 HP / 28 Def / 232 SpD
Sassy Nature
IVs: 0 Spe
- Surf
- U-turn
- Roost
- Hurricane

Barraskewda @ Choice Band
Ability: Swift Swim
Tera Type: Water
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Liquidation
- Flip Turn
- Aqua Jet
- Close Combat

Iron Treads @ Booster Energy
Ability: Quark Drive
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Rapid Spin
- Earth Power
- Steel Beam
- Stealth Rock

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Calm Mind
- Thunderbolt
- Moonblast
- Encore

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Fairy
EVs: 212 HP / 252 Atk / 44 Spe
Adamant Nature
- Swords Dance
- Sucker Punch
- Kowtow Cleave
- Iron Head""",
    """
Ninetales-Alola (F) @ Light Clay
Ability: Snow Warning
Tera Type: Water
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Aurora Veil
- Encore
- Freeze-Dry
- Moonblast

Tyranitar @ Weakness Policy
Ability: Sand Stream
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Stone Edge
- Knock Off
- Ice Punch

Gouging Fire @ Leftovers
Ability: Protosynthesis
Tera Type: Poison
EVs: 252 HP / 72 Atk / 104 Def / 16 SpD / 64 Spe
Impish Nature
- Dragon Dance
- Flare Blitz
- Breaking Swipe
- Morning Sun

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Steel
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Calm Mind
- Moonblast
- Psyshock
- Encore

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Poison
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Bulk Up
- Headlong Rush
- Ice Spinner
- Rapid Spin

Deoxys-Speed @ Eject Pack
Ability: Pressure
Tera Type: Fighting
EVs: 64 Atk / 192 SpA / 252 Spe
Naive Nature
- Stealth Rock
- Spikes
- Superpower
- Psycho Boost""",
    """
Slowking-Galar @ Black Sludge
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 244 Def / 16 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Sludge Bomb
- Trick
- Thunder Wave
- Chilly Reception

Ogerpon-Wellspring (F) @ Wellspring Mask
Ability: Water Absorb
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Knock Off
- U-turn

Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Grass
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Flamethrower
- Scorching Sands
- Weather Ball
- Roost

Weezing-Galar @ Heavy-Duty Boots
Ability: Neutralizing Gas
Tera Type: Water
EVs: 248 HP / 244 Def / 16 Spe
Bold Nature
IVs: 0 Atk
- Strange Steam
- Toxic
- Defog
- Pain Split

Dragapult @ Spell Tag
Ability: Infiltrator
Tera Type: Fighting
EVs: 44 HP / 92 Atk / 248 SpA / 124 Spe
Hasty Nature
- Shadow Ball
- Dragon Darts
- Substitute
- Tera Blast

Ting-Lu @ Rocky Helmet
Ability: Vessel of Ruin
Tera Type: Water
EVs: 248 HP / 16 Def / 244 SpD
Impish Nature
- Earthquake
- Ruination
- Stealth Rock
- Whirlwind""",
    """
Darkrai @ Heavy-Duty Boots
Ability: Bad Dreams
Tera Type: Poison
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Dark Pulse
- Sludge Bomb
- Ice Beam
- Knock Off

Raging Bolt @ Booster Energy
Ability: Protosynthesis
Tera Type: Fairy
EVs: 112 HP / 252 SpA / 144 Spe
Modest Nature
IVs: 20 Atk
- Thunderbolt
- Dragon Pulse
- Thunderclap
- Calm Mind

Ting-Lu @ Rocky Helmet
Ability: Vessel of Ruin
Tera Type: Ghost
EVs: 240 HP / 16 Def / 64 SpD / 188 Spe
Impish Nature
- Earthquake
- Ruination
- Spikes
- Taunt

Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Fairy
EVs: 248 HP / 72 Atk / 156 SpD / 32 Spe
Brave Nature
- Brave Bird
- U-turn
- Roost
- Will-O-Wisp

Tinkaton @ Air Balloon
Ability: Pickpocket
Tera Type: Water
EVs: 248 HP / 72 Def / 96 SpD / 92 Spe
Jolly Nature
- Stealth Rock
- Gigaton Hammer
- Knock Off
- Encore

Dragapult (F) @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Fairy
EVs: 96 Atk / 160 SpA / 252 Spe
Hasty Nature
- Hex
- Dragon Darts
- Thunder Wave
- U-turn""",
]

GEN4TEAMS = [
    """
Empoleon @ Focus Sash
Ability: Torrent
EVs: 4 Atk / 252 SpA / 252 Spe
Modest Nature
- Stealth Rock
- Hydro Pump
- Aqua Jet
- Grass Knot

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 56 HP / 232 SpA / 220 Spe
Timid Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Calm Mind
- Psychic
- Grass Knot
- Hidden Power [Fire]

Starmie @ Colbur Berry
Ability: Natural Cure
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Thunderbolt
- Ice Beam
- Rapid Spin

Bronzong @ Macho Brace
Ability: Levitate
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Trick Room
- Gyro Ball
- Earthquake
- Explosion

Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 128 HP / 252 Atk / 128 Spe
Adamant Nature
- Crunch
- Pursuit
- Stone Edge
- Superpower

Dragonite @ Lum Berry
Ability: Inner Focus
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Outrage
- Fire Punch
- Extreme Speed""",
    """
Metagross @ Focus Sash
Ability: Clear Body
EVs: 80 HP / 252 Atk / 176 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Bullet Punch
- Explosion

Scizor @ Leftovers
Ability: Technician
EVs: 248 HP / 56 Atk / 12 Def / 192 SpD
Adamant Nature
- Bullet Punch
- Swords Dance
- Brick Break
- Roost

Tyranitar @ Custap Berry
Ability: Sand Stream
EVs: 252 HP / 108 Atk / 148 SpD
Adamant Nature
- Crunch
- Pursuit
- Earthquake
- Stealth Rock

Infernape @ Expert Belt
Ability: Blaze
EVs: 56 Atk / 252 SpA / 200 Spe
Naive Nature
- Fire Blast
- Hidden Power [Ice]
- Close Combat
- Grass Knot

Rotom-Wash @ Choice Scarf
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 2 Atk / 30 Def
- Thunderbolt
- Shadow Ball
- Hidden Power [Ice]
- Trick

Latias @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Trick
- Draco Meteor
- Grass Knot
- Surf""",
    """
Zapdos @ Leftovers
Ability: Pressure
Shiny: Yes
EVs: 248 HP / 248 Def / 12 SpD
Bold Nature
IVs: 2 Atk / 30 Def
- Discharge
- Heat Wave
- Hidden Power [Ice]
- Roost

Heatran (M) @ Choice Scarf
Ability: Flash Fire
Shiny: Yes
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Fire Blast
- Earth Power
- Hidden Power [Ice]
- Explosion

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 68 Atk / 20 Def / 168 SpD
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Protect

Blissey @ Leftovers
Ability: Natural Cure
Shiny: Yes
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Stealth Rock
- Seismic Toss
- Soft-Boiled
- Flamethrower

Starmie @ Leftovers
Ability: Natural Cure
Shiny: Yes
EVs: 248 HP / 44 Def / 216 Spe
Timid Nature
IVs: 0 Atk
- Surf
- Psychic
- Recover
- Rapid Spin

Breloom (M) @ Toxic Orb
Ability: Poison Heal
Shiny: Yes
EVs: 204 HP / 252 Atk / 20 SpD / 32 Spe
Adamant Nature
- Spore
- Superpower
- Seed Bomb
- Mach Punch""",
    """
Azelf @ Lum Berry
Ability: Levitate
EVs: 200 HP / 224 Def / 24 SpD / 60 Spe
Jolly Nature
- Taunt
- Stealth Rock
- Thunder Wave
- Explosion

Gyarados (M) @ Lum Berry
Ability: Intimidate
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Waterfall
- Earthquake
- Ice Fang

Zapdos @ Life Orb
Ability: Pressure
Shiny: Yes
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 2 Atk / 30 SpA
- Agility
- Thunderbolt
- Heat Wave
- Hidden Power [Grass]

Metagross @ Iron Ball
Ability: Clear Body
Shiny: Yes
EVs: 248 HP / 136 Atk / 20 Def / 76 SpD / 28 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Trick
- Explosion

Tyranitar (M) @ Shuca Berry
Ability: Sand Stream
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Stone Edge
- Earthquake
- Ice Punch

Jirachi @ Shuca Berry
Ability: Serene Grace
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 3 Atk / 30 SpA / 30 SpD
- Calm Mind
- Psychic
- Grass Knot
- Hidden Power [Ground]""",
    """
Roserade @ Focus Sash
Ability: Poison Point
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 3 Atk / 30 SpA / 30 SpD
- Sleep Powder
- Leaf Storm
- Toxic Spikes
- Hidden Power [Ground]

Heatran @ Leftovers
Ability: Flash Fire
EVs: 240 HP / 24 Atk / 244 SpD
Sassy Nature
- Stealth Rock
- Lava Plume
- Protect
- Explosion

Jirachi @ Choice Scarf
Ability: Serene Grace
EVs: 16 HP / 212 Atk / 28 SpD / 252 Spe
Jolly Nature
- U-turn
- Iron Head
- Fire Punch
- Trick

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 120 SpA / 136 Spe
Timid Nature
IVs: 0 Atk
- Calm Mind
- Hydro Pump
- Ice Beam
- Substitute

Latias @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Trick
- Draco Meteor
- Surf
- Healing Wish

Rotom-Heat @ Leftovers
Ability: Levitate
EVs: 112 HP / 192 SpA / 204 Spe
Timid Nature
IVs: 0 Atk
- Substitute
- Charge Beam
- Shadow Ball
- Thunderbolt""",
    """
Skarmory @ Lum Berry
Ability: Keen Eye
EVs: 248 HP / 48 Atk / 20 Def / 8 SpD / 184 Spe
Jolly Nature
- Taunt
- Stealth Rock
- Spikes
- Drill Peck

Flygon @ Life Orb
Ability: Levitate
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Roost
- Fire Blast
- Draco Meteor
- Earthquake

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 172 Atk / 72 SpA / 12 Spe
Lonely Nature
- Substitute
- Focus Punch
- Ice Beam
- Crunch

Gyarados @ Lum Berry
Ability: Intimidate
EVs: 40 HP / 252 Atk / 216 Spe
Adamant Nature
- Dragon Dance
- Waterfall
- Ice Fang
- Earthquake

Lucario @ Life Orb
Ability: Inner Focus
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Swords Dance
- Close Combat
- Extreme Speed
- Bullet Punch

Rotom-Heat @ Choice Scarf
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Thunderbolt
- Shadow Ball
- Overheat
- Trick""",
    """
Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 196 Def / 60 SpD
Impish Nature
- Iron Head
- Body Slam
- Wish
- Protect

Flygon (M) @ Choice Scarf
Ability: Levitate
Shiny: Yes
EVs: 244 Atk / 12 Def / 252 Spe
Jolly Nature
- U-turn
- Earthquake
- Outrage
- Toxic

Milotic (F) @ Leftovers
Ability: Marvel Scale
Shiny: Yes
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
IVs: 0 Atk
- Surf
- Ice Beam
- Recover
- Haze

Gliscor (M) @ Leftovers
Ability: Hyper Cutter
Shiny: Yes
EVs: 252 HP / 40 Def / 216 Spe
Impish Nature
- Taunt
- Earthquake
- Wing Attack
- Roost

Clefable (F) @ Leftovers
Ability: Magic Guard
Shiny: Yes
EVs: 252 HP / 16 Def / 240 SpD
Calm Nature
- Stealth Rock
- Seismic Toss
- Knock Off
- Soft-Boiled

Magnezone @ Leftovers
Ability: Magnet Pull
EVs: 32 HP / 4 Def / 252 SpA / 220 Spe
Modest Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Thunderbolt
- Hidden Power [Fire]
- Magnet Rise
- Protect""",
    """
Hippowdon @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 8 Def / 252 SpD
Careful Nature
- Stealth Rock
- Earthquake
- Slack Off
- Roar

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 248 HP / 216 Def / 44 Spe
Impish Nature
IVs: 0 Atk
- Taunt
- Spikes
- Roost
- Whirlwind

Clefable @ Leftovers
Ability: Magic Guard
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Seismic Toss
- Knock Off
- Stealth Rock
- Soft-Boiled

Starmie @ Leftovers
Ability: Natural Cure
EVs: 248 HP / 204 Def / 56 Spe
Bold Nature
IVs: 0 Atk
- Hydro Pump
- Psychic
- Recover
- Rapid Spin

Gliscor @ Leftovers
Ability: Hyper Cutter
EVs: 248 HP / 44 Def / 216 Spe
Jolly Nature
- Taunt
- Earthquake
- Ice Fang
- Roost

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 36 Atk / 44 Def / 176 SpD
Careful Nature
- Wish
- Protect
- Iron Head
- Body Slam""",
    """
Tyranitar (M) @ Choice Scarf
Ability: Sand Stream
Shiny: Yes
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Crunch
- Thunder Wave
- Stealth Rock
- Pursuit

Skarmory @ Shed Shell
Ability: Keen Eye
EVs: 248 HP / 244 Def / 16 Spe
Impish Nature
- Brave Bird
- Roost
- Spikes
- Whirlwind

Quagsire (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 100 Def / 156 SpD
Impish Nature
- Earthquake
- Encore
- Recover
- Toxic

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 180 HP / 180 Atk / 148 Spe
Jolly Nature
- Iron Head
- Fire Punch
- Body Slam
- Protect

Latias @ Leftovers
Ability: Levitate
EVs: 252 HP / 160 Def / 96 Spe
Bold Nature
- Ice Beam
- Earthquake
- Thunder Wave
- Recover

Clefable (F) @ Leftovers
Ability: Magic Guard
EVs: 252 HP / 56 Def / 200 SpD
Calm Nature
- Seismic Toss
- Thunder Wave
- Soft-Boiled
- Knock Off""",
    """
Rotom-Frost @ Leftovers
Ability: Levitate
EVs: 244 HP / 108 SpA / 68 SpD / 88 Spe
Modest Nature
IVs: 0 Atk
- Thunderbolt
- Blizzard
- Will-O-Wisp
- Pain Split

Skarmory (M) @ Leftovers
Ability: Sturdy
EVs: 252 HP / 192 SpD / 64 Spe
Careful Nature
IVs: 0 Atk
- Spikes
- Taunt
- Roost
- Whirlwind

Swampert (M) @ Leftovers
Ability: Torrent
EVs: 204 HP / 104 Atk / 112 Def / 88 SpD
Adamant Nature
- Stealth Rock
- Earthquake
- Waterfall
- Ice Punch

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 156 HP / 176 SpA / 176 Spe
Modest Nature
IVs: 3 Atk / 30 SpA / 30 SpD
- Draco Meteor
- Thunder
- Hidden Power [Ground]
- Calm Mind

Latias (F) @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Dragon Pulse
- Surf
- Sleep Talk

Lucario (M) @ Choice Scarf
Ability: Inner Focus
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Close Combat
- Ice Punch
- Thunder Punch
- Crunch""",
    """
Bronzong @ Leftovers
Ability: Levitate
EVs: 252 HP / 28 Atk / 152 Def / 76 SpD
Relaxed Nature
IVs: 0 Spe
- Stealth Rock
- Gyro Ball
- Protect
- Toxic

Skarmory (M) @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
IVs: 0 Atk
- Spikes
- Counter
- Roost
- Whirlwind

Clefable (F) @ Leftovers
Ability: Magic Guard
EVs: 252 HP / 72 Def / 184 SpD
Careful Nature
IVs: 30 Spe
- Seismic Toss
- Knock Off
- Thunder Wave
- Soft-Boiled

Latias @ Leftovers
Ability: Levitate
EVs: 204 HP / 120 SpA / 184 Spe
Timid Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Calm Mind
- Dragon Pulse
- Hidden Power [Fire]
- Recover

Flygon (M) @ Life Orb
Ability: Levitate
EVs: 32 Atk / 224 SpA / 252 Spe
Naive Nature
- Earthquake
- Draco Meteor
- Fire Blast
- Roost

Rotom-Frost @ Leftovers
Ability: Levitate
EVs: 252 HP / 248 Def / 8 Spe
Bold Nature
IVs: 0 Atk
- Discharge
- Shadow Ball
- Rest
- Sleep Talk""",
    """
Zapdos @ Leftovers
Ability: Pressure
EVs: 248 HP / 216 Def / 12 SpD / 32 Spe
Bold Nature
IVs: 2 Atk / 30 Def
- Thunderbolt
- Hidden Power [Ice]
- Roost
- Substitute

Forretress @ Leftovers
Ability: Sturdy
EVs: 248 HP / 8 Def / 252 SpD
Careful Nature
- Spikes
- Rapid Spin
- Rest
- Payback

Latias @ Leftovers
Ability: Levitate
EVs: 204 HP / 120 SpA / 184 Spe
Timid Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Calm Mind
- Dragon Pulse
- Hidden Power [Fire]
- Recover

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 248 HP / 168 Def / 60 SpD / 32 Spe
Impish Nature
- Wish
- Protect
- Body Slam
- Iron Head

Swampert @ Leftovers
Ability: Torrent
EVs: 248 HP / 216 Def / 40 SpD / 4 Spe
Relaxed Nature
- Stealth Rock
- Earthquake
- Ice Beam
- Roar

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
IVs: 0 Atk
- Seismic Toss
- Ice Beam
- Soft-Boiled
- Heal Bell""",
    """
Swampert @ Leftovers
Ability: Torrent
EVs: 36 Def / 252 SpA / 220 Spe
Modest Nature
IVs: 0 Atk
- Stealth Rock
- Hydro Pump
- Earth Power
- Ice Beam

Latias (F) @ Choice Scarf
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Thunder Wave
- Trick
- Healing Wish

Jirachi @ Lum Berry
Ability: Serene Grace
EVs: 40 Atk / 252 SpA / 216 Spe
Hasty Nature
- Thunder
- Iron Head
- Psychic
- Fire Punch

Metagross @ Iron Ball
Ability: Clear Body
EVs: 236 HP / 84 Atk / 8 Def / 152 SpD / 28 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Trick
- Explosion

Tyranitar @ Lum Berry
Ability: Sand Stream
EVs: 152 SpA / 104 SpD / 252 Spe
Hasty Nature
- Crunch
- Fire Blast
- Superpower
- Toxic

Gengar @ Black Sludge
Ability: Levitate
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Shadow Ball
- Focus Blast
- Will-O-Wisp
- Explosion""",
]

TEAM1 = """
Bronzong @ Lum Berry
Ability: Heatproof
EVs: 248 HP / 252 Atk / 4 Def / 4 SpD
Brave Nature
IVs: 0 Spe
- Stealth Rock
- Gyro Ball
- Earthquake
- Explosion

Gengar @ Life Orb
Ability: Levitate
EVs: 4 Atk / 252 SpA / 252 Spe
Naive Nature
- Shadow Ball
- Focus Blast
- Explosion
- Hidden Power [Fire]

Tyranitar (M) @ Choice Band
Ability: Sand Stream
EVs: 252 Atk / 44 SpD / 212 Spe
Adamant Nature
- Stone Edge
- Crunch
- Pursuit
- Superpower

Kingdra @ Choice Specs
Ability: Swift Swim
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hydro Pump
- Draco Meteor
- Surf
- Dragon Pulse

Flygon (M) @ Choice Scarf
Ability: Levitate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- U-turn
- Outrage
- Earthquake
- Thunder Punch

Lucario @ Life Orb
Ability: Inner Focus
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Close Combat
- Swords Dance
- Bullet Punch
- Extreme Speed"""

TEAM2 = """
Bronzong @ Lum Berry
Ability: Heatproof
EVs: 248 HP / 252 Atk / 8 SpD
Brave Nature
IVs: 0 Spe
- Gyro Ball
- Stealth Rock
- Earthquake
- Explosion

Dragonite (M) @ Choice Band
Ability: Inner Focus
EVs: 48 HP / 252 Atk / 208 Spe
Adamant Nature
- Outrage
- Dragon Claw
- Extreme Speed
- Earthquake

Mamoswine (M) @ Life Orb
Ability: Oblivious
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Ice Shard
- Earthquake
- Stone Edge
- Superpower

Magnezone @ Leftovers
Ability: Magnet Pull
EVs: 140 HP / 252 SpA / 116 Spe
Modest Nature
IVs: 2 Atk / 30 SpA / 30 Spe
- Thunderbolt
- Thunder Wave
- Substitute
- Hidden Power [Fire]

Flygon @ Choice Scarf
Ability: Levitate
EVs: 252 Atk / 6 Def / 252 Spe
Adamant Nature
- Outrage
- Earthquake
- Stone Edge
- U-turn

Kingdra (M) @ Chesto Berry
Ability: Swift Swim
EVs: 144 HP / 160 Atk / 40 SpD / 164 Spe
Adamant Nature
- Dragon Dance
- Waterfall
- Outrage
- Rest"""
