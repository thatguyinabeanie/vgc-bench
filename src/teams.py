import random

from poke_env.teambuilder import Teambuilder


class RandomTeamBuilder(Teambuilder):
    teams: list[str]

    def __init__(self):
        self.teams = []
        for team in TEAMS:
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
