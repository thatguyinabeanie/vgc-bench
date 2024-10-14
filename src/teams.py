# teams from https://docs.google.com/spreadsheets/d/1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw/edit?gid=511126880#gid=511126880

import random
from subprocess import run

from poke_env.teambuilder import Teambuilder


class RandomTeamBuilder(Teambuilder):
    teams: list[str]

    def __init__(self, num_teams: int, battle_format: str):
        self.teams = []
        for team in TEAMS[battle_format][:num_teams]:
            result = run(
                ["node", "pokemon-showdown", "validate-team", battle_format],
                input=f'"{team[1:]}"'.encode(),
                cwd="pokemon-showdown",
                capture_output=True,
            )
            if result.returncode == 1:
                print(f"team {TEAMS[battle_format].index(team)}: {result.stderr.decode()}")
            else:
                parsed_team = self.parse_showdown_team(team)
                packed_team = self.join_team(parsed_team)
                self.teams.append(packed_team)

    def yield_team(self) -> str:
        return random.choice(self.teams)


TEAMS = {
    "gen9vgc2024regh": [
        """
Maradona (Iron Hands) @ Assault Vest
Ability: Quark Drive
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 140 HP / 76 Atk / 4 Def / 252 SpD / 36 Spe
Adamant Nature
- Fake Out
- Close Combat
- Wild Charge
- Volt Switch

Shrom41Mor (Amoonguss) (F) @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Steel
EVs: 244 HP / 108 Def / 156 SpD
Calm Nature
IVs: 0 Atk / 0 Spe
- Spore
- Rage Powder
- Pollen Puff
- Protect

Beaker (Pelipper) (F) @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Flying
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
IVs: 0 Atk
- Hurricane
- Hydro Pump
- Tailwind
- Protect

Torqoise (Palafin) (F) @ Mystic Water
Ability: Zero to Hero
Level: 50
Tera Type: Water
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Jet Punch
- Wave Crash
- Haze
- Protect

BlockBuster (Baxcalibur) (F) @ Dragon Fang
Ability: Thermal Exchange
Level: 50
Tera Type: Poison
EVs: 84 HP / 252 Atk / 4 Def / 108 SpD / 60 Spe
Adamant Nature
- Glaive Rush
- Ice Shard
- Icicle Crash
- Protect

ShiningArmor (Dragonite) (F) @ Lum Berry
Ability: Multiscale
Level: 50
Tera Type: Flying
EVs: 252 HP / 204 Atk / 20 Def / 4 SpD / 28 Spe
Adamant Nature
- Extreme Speed
- Tera Blast
- Ice Spinner
- Protect
""",
        """
Tatsugiri @ Choice Scarf
Ability: Commander
Level: 50
Tera Type: Steel
EVs: 180 Def / 124 SpA / 204 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Muddy Water
- Dragon Pulse
- Icy Wind

Dondozo @ Leftovers
Ability: Unaware
Level: 50
Tera Type: Steel
EVs: 76 HP / 84 Atk / 12 Def / 124 SpD / 212 Spe
Adamant Nature
- Wave Crash
- Earthquake
- Heavy Slam
- Yawn

Flutter Mane @ Life Orb
Ability: Protosynthesis
Level: 50
Tera Type: Grass
EVs: 132 HP / 156 Def / 68 SpA / 4 SpD / 148 Spe
Timid Nature
IVs: 0 Atk
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Protect

Iron Moth @ Booster Energy
Ability: Quark Drive
Level: 50
Tera Type: Grass
EVs: 156 HP / 4 Def / 108 SpA / 4 SpD / 236 Spe
Timid Nature
IVs: 0 Atk
- Flamethrower
- Acid Spray
- Energy Ball
- Protect

Dragonite @ Choice Band
Ability: Multiscale
Level: 50
Tera Type: Normal
EVs: 196 HP / 252 Atk / 4 Def / 4 SpD / 52 Spe
Adamant Nature
- Extreme Speed
- Aerial Ace
- Ice Spinner
- Stomping Tantrum

Arcanine @ Safety Goggles
Ability: Intimidate
Tera Type: Normal
EVs: 252 HP / 84 Atk / 84 Def / 12 SpD / 76 Spe
Adamant Nature
- Flare Blitz
- Extreme Speed
- Will-O-Wisp
- Protect""",
        """
Roaring Moon @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Flying
EVs: 252 HP / 4 Atk / 12 Def / 52 SpD / 188 Spe
Adamant Nature
- Acrobatics
- Dragon Dance
- Jaw Lock
- Roost

Gholdengo @ Covert Cloak
Ability: Good as Gold
Level: 50
Tera Type: Water
EVs: 252 HP / 4 Def / 76 SpA / 100 SpD / 76 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Nasty Plot
- Shadow Ball
- Protect

Iron Bundle @ Focus Sash
Ability: Quark Drive
Level: 50
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Freeze-Dry
- Icy Wind
- Protect

Iron Hands @ Assault Vest
Ability: Quark Drive
Level: 50
Tera Type: Grass
EVs: 180 HP / 4 Atk / 4 Def / 252 SpD / 68 Spe
Adamant Nature
- Fake Out
- Close Combat
- Wild Charge
- Volt Switch

Maushold-Four @ Safety Goggles
Ability: Friend Guard
Level: 50
Tera Type: Ghost
EVs: 252 HP / 28 Def / 228 Spe
Timid Nature
IVs: 0 Atk
- Super Fang
- Follow Me
- Baby-Doll Eyes
- Protect

Arcanine (F) @ Sitrus Berry
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 244 HP / 4 Atk / 4 Def / 164 SpD / 92 Spe
Careful Nature
- Flare Blitz
- Will-O-Wisp
- Snarl
- Extreme Speed""",
        """
Feelin U? (Flutter Mane) @ Choice Specs
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Mystical Fire

Munch (Armarouge) (M) @ Life Orb
Ability: Flash Fire
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 228 HP / 252 SpA / 28 SpD
Modest Nature
IVs: 0 Atk
- Expanding Force
- Armor Cannon
- Trick Room
- Protect

Ice Spice (Indeedee-F) @ Psychic Seed
Ability: Psychic Surge
Level: 50
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
IVs: 2 Atk / 3 Spe
- Follow Me
- Dazzling Gleam
- Helping Hand
- Trick Room

Quach (Great Tusk) @ Focus Sash
Ability: Protosynthesis
Level: 50
Shiny: Yes
Tera Type: Ground
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Headlong Rush
- Close Combat
- Taunt
- Protect

God Willing (Iron Bundle) @ Booster Energy
Ability: Quark Drive
Level: 50
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Icy Wind
- Hydro Pump
- Freeze-Dry
- Protect

Gucci Hamish (Hydreigon) @ Safety Goggles
Ability: Levitate
Level: 50
Tera Type: Psychic
EVs: 108 HP / 36 Def / 164 SpA / 4 SpD / 196 Spe
Timid Nature
IVs: 0 Atk
- Snarl
- Draco Meteor
- Dark Pulse
- Tailwind""",
        """
Maroon (Talonflame) (F) @ Covert Cloak
Ability: Gale Wings
Level: 50
Tera Type: Ghost
EVs: 132 HP / 76 Atk / 44 Def / 4 SpD / 252 Spe
Jolly Nature
IVs: 22 SpA
- Brave Bird
- Will-O-Wisp
- Taunt
- Tailwind

Anti-Hero (Dondozo) (M) @ Leftovers
Ability: Unaware
Level: 50
Tera Type: Steel
EVs: 60 Atk / 244 SpD / 204 Spe
Adamant Nature
IVs: 9 SpA
- Earthquake
- Order Up
- Substitute
- Protect

Bejeweled (Gholdengo) @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 244 HP / 76 Def / 140 SpA / 28 SpD / 20 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Thunderbolt
- Power Gem

Mastermind (Pawmot) (M) @ Focus Sash
Ability: Natural Cure
Level: 50
Tera Type: Electric
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Double Shock
- Close Combat
- Revival Blessing
- Fake Out

On Your Own (Tatsugiri) (M) @ Choice Scarf
Ability: Commander
Level: 50
Tera Type: Water
EVs: 20 HP / 60 Def / 172 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Muddy Water
- Draco Meteor
- Icy Wind
- Sleep Talk

Question...? (Arcanine) (F) @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Dark
EVs: 244 HP / 36 Atk / 4 Def / 4 SpD / 220 Spe
Jolly Nature
- Flare Blitz
- Will-O-Wisp
- Snarl
- Protect""",
        """
Iron Bundle @ Covert Cloak
Ability: Quark Drive
Level: 50
Tera Type: Ice
EVs: 36 HP / 4 Def / 180 SpA / 36 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Icy Wind
- Hydro Pump
- Freeze-Dry
- Protect

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Steel
EVs: 252 HP / 164 Def / 36 SpA / 4 SpD / 52 Spe
Modest Nature
IVs: 1 Atk
- Shadow Ball
- Protect
- Trick Room
- Dazzling Gleam

Great Tusk @ Focus Sash
Ability: Protosynthesis
Level: 50
Tera Type: Ground
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Earthquake
- Close Combat
- Protect

Talonflame (M) @ Safety Goggles
Ability: Gale Wings
Level: 50
Tera Type: Ghost
EVs: 204 HP / 44 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Brave Bird
- Taunt
- Will-O-Wisp
- Tailwind

Kingambit (M) @ Assault Vest
Ability: Defiant
Level: 50
Tera Type: Flying
EVs: 244 HP / 36 Atk / 4 Def / 4 SpD / 220 Spe
Adamant Nature
- Sucker Punch
- Tera Blast
- Assurance
- Iron Head

Glimmora (M) @ Life Orb
Ability: Toxic Debris
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 20 HP / 236 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Sludge Wave
- Power Gem
- Spiky Shield
- Earth Power""",
        """
谜拟丘 (Mimikyu) @ Life Orb
Ability: Disguise
Level: 50
Tera Type: Grass
EVs: 68 HP / 252 Atk / 188 Spe
Jolly Nature
- Play Rough
- Shadow Sneak
- Curse
- Protect

盐石巨灵 (Garganacl) @ Leftovers
Ability: Purifying Salt
Level: 50
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 SpD
Impish Nature
- Salt Cure
- Protect
- Recover
- Wide Guard

铁臂膀 (Iron Hands) @ Assault Vest
Ability: Quark Drive
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 76 HP / 156 Atk / 4 Def / 252 SpD / 20 Spe
Adamant Nature
- Fake Out
- Volt Switch
- Wild Charge
- Drain Punch

轰鸣月 (Roaring Moon) @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Flying
EVs: 212 HP / 4 Atk / 36 Def / 4 SpD / 252 Spe
Adamant Nature
- Acrobatics
- Throat Chop
- Dragon Dance
- Protect

败露球菇 (Amoonguss) @ Covert Cloak
Ability: Regenerator
Level: 50
Shiny: Yes
Tera Type: Dark
EVs: 236 HP / 156 Def / 116 SpD
Bold Nature
IVs: 24 Spe
- Protect
- Rage Powder
- Pollen Puff
- Spore

铁包袱 (Iron Bundle) @ Focus Sash
Ability: Quark Drive
Level: 50
Shiny: Yes
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Protect
- Freeze-Dry
- Encore
- Hydro Pump""",
        """
Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
Tera Type: Ghost
EVs: 252 HP / 4 Atk / 132 Def / 116 SpD / 4 Spe
Careful Nature
- Spirit Break
- Light Screen
- Reflect
- Misty Terrain

Arcanine @ Assault Vest
Ability: Intimidate
Level: 50
Tera Type: Fire
EVs: 220 HP / 156 Atk / 12 Def / 28 SpD / 92 Spe
Adamant Nature
- Flare Blitz
- Snarl
- Extreme Speed
- Psychic Fangs

Iron Hands @ Sitrus Berry
Ability: Quark Drive
Level: 50
Tera Type: Grass
EVs: 84 HP / 140 Atk / 20 Def / 220 SpD / 44 Spe
Adamant Nature
- Drain Punch
- Wild Charge
- Swords Dance
- Protect

Gastrodon-East @ Covert Cloak
Ability: Storm Drain
Level: 50
Tera Type: Fire
EVs: 180 HP / 116 Def / 140 SpA / 68 SpD / 4 Spe
Quiet Nature
IVs: 0 Atk / 0 Spe
- Earth Power
- Muddy Water
- Recover
- Protect

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Rock
EVs: 236 HP / 116 Def / 60 SpA / 44 SpD / 52 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Moonblast
- Shadow Ball
- Power Gem

Talonflame @ Sharp Beak
Ability: Gale Wings
Level: 50
Tera Type: Ghost
EVs: 4 HP / 196 Atk / 4 Def / 252 Spe
Jolly Nature
- Will-O-Wisp
- Brave Bird
- Tailwind
- Taunt""",
    ],
    "gen9ou": [
        """
Whiscash @ Leftovers
Ability: Oblivious
Tera Type: Poison
EVs: 252 HP / 252 SpA / 4 Spe
Modest Nature
IVs: 0 Atk
- Stealth Rock
- Spikes
- Hydro Pump
- Earth Power

Duraludon @ Eviolite
Ability: Heavy Metal
Tera Type: Ghost
EVs: 200 HP / 252 Def / 4 SpA / 52 Spe
Bold Nature
- Iron Defense
- Body Press
- Flash Cannon
- Dragon Tail

Zamazenta @ Light Clay
Ability: Dauntless Shield
Tera Type: Steel
EVs: 252 HP / 152 SpD / 104 Spe
Timid Nature
IVs: 0 Atk
- Reflect
- Light Screen
- Body Press
- Roar

Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Flying
EVs: 88 HP / 188 SpA / 232 Spe
Timid Nature
- Hurricane
- Flamethrower
- U-turn
- Roost

Great Tusk @ Lum Berry
Ability: Protosynthesis
Tera Type: Ice
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Bulk Up
- Earthquake
- Ice Spinner
- Rapid Spin

Ogerpon (F) @ Heavy-Duty Boots
Ability: Defiant
Tera Type: Grass
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Swords Dance
- Ivy Cudgel
- Knock Off
- Rock Tomb""",
        """
Landorus-Therian @ Leftovers
Ability: Intimidate
Shiny: Yes
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Smack Down
- Substitute
- Swords Dance

Ribombee @ Focus Sash
Ability: Shield Dust
Shiny: Yes
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Moonblast
- Stun Spore
- Skill Swap
- Sticky Web

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Dazzling Gleam
- Nasty Plot

Ogerpon-Wellspring @ Wellspring Mask
Ability: Water Absorb
Tera Type: Water
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Power Whip
- Ivy Cudgel
- Encore
- Swords Dance

Iron Treads @ Booster Energy
Ability: Quark Drive
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Earth Power
- Steel Beam
- Rapid Spin
- Stealth Rock

Darkrai @ Expert Belt
Ability: Bad Dreams
Shiny: Yes
Tera Type: Poison
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Dark Pulse
- Sludge Bomb
- Ice Beam
- Focus Blast""",
        """
Cinderace @ Focus Sash
Ability: Blaze
Tera Type: Fighting
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Pyro Ball
- Reversal
- Sucker Punch

Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Grass
EVs: 232 HP / 20 Def / 252 Spe
Timid Nature
- Stealth Rock
- Earth Power
- Taunt
- U-turn

Hatterene @ Rocky Helmet
Ability: Magic Bounce
Tera Type: Steel
EVs: 248 HP / 200 Def / 60 Spe
Bold Nature
- Psychic Noise
- Draining Kiss
- Nuzzle
- Healing Wish

Darkrai @ Roseli Berry
Ability: Bad Dreams
Tera Type: Poison
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Nasty Plot
- Dark Pulse
- Focus Blast
- Sludge Bomb

Kyurem @ Loaded Dice
Ability: Pressure
Tera Type: Electric
EVs: 56 HP / 252 Atk / 4 Def / 196 Spe
Jolly Nature
- Dragon Dance
- Substitute
- Icicle Spear
- Tera Blast

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Dark
EVs: 96 Atk / 160 SpA / 252 Spe
Naive Nature
- Destiny Bond
- Moonblast
- Knock Off
- Close Combat""",
        """
Weezing-Galar @ Terrain Extender
Ability: Misty Surge
Tera Type: Grass
EVs: 76 HP / 252 SpA / 180 Spe
Modest Nature
IVs: 0 Atk
- Strange Steam
- Sludge Wave
- Fire Blast
- Taunt

Ogerpon-Wellspring (F) @ Wellspring Mask
Ability: Water Absorb
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Ivy Cudgel
- Power Whip
- Play Rough

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 252 HP / 72 Def / 184 Spe
Bold Nature
IVs: 0 Atk
- Nasty Plot
- Shadow Ball
- Dazzling Gleam
- Recover

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Bulk Up
- Headlong Rush
- Ice Spinner
- Rapid Spin

Roaring Moon @ Booster Energy
Ability: Protosynthesis
Shiny: Yes
Tera Type: Ground
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Knock Off
- Acrobatics
- Earthquake

Glimmora @ Red Card
Ability: Toxic Debris
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Stealth Rock
- Mortal Spin
- Power Gem
- Earth Power""",
        """
Rillaboom @ Terrain Extender
Ability: Grassy Surge
Tera Type: Steel
EVs: 200 HP / 252 Atk / 56 Spe
Adamant Nature
- Grassy Glide
- Knock Off
- Low Kick
- U-turn

Bellossom (F) @ Grassy Seed
Ability: Chlorophyll
Shiny: Yes
Tera Type: Fire
EVs: 80 HP / 4 Def / 232 SpA / 192 Spe
Timid Nature
- Giga Drain
- Tera Blast
- Quiver Dance
- Strength Sap

Glimmora @ Focus Sash
Ability: Toxic Debris
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Mortal Spin
- Dazzling Gleam
- Mud Shot
- Stealth Rock

Hawlucha @ Grassy Seed
Ability: Unburden
Tera Type: Ground
EVs: 72 HP / 252 Atk / 60 SpD / 124 Spe
Adamant Nature
- Swords Dance
- Close Combat
- Acrobatics
- Encore

Hatterene @ Grassy Seed
Ability: Magic Bounce
Tera Type: Steel
EVs: 252 HP / 192 Def / 64 Spe
Bold Nature
- Calm Mind
- Draining Kiss
- Stored Power
- Nuzzle

Roaring Moon @ Booster Energy
Ability: Protosynthesis
Tera Type: Flying
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Knock Off
- Acrobatics
- Brick Break""",
    ],
}
