# teams from https://docs.google.com/spreadsheets/d/1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw/edit?gid=1168048410#gid=1168048410

import random
from subprocess import run

from poke_env.teambuilder import Teambuilder


class RandomTeamBuilder(Teambuilder):
    teams: list[str]

    def __init__(self, teams: list[int], battle_format: str):
        self.teams = []
        for team in [TEAMS[battle_format][t] for t in teams]:
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
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Ghost
EVs: 188 HP / 164 Atk / 4 Def / 108 SpD / 44 Spe
Adamant Nature
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot

Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hurricane
- Weather Ball
- Tailwind
- Protect

Basculegion (M) @ Mystic Water
Ability: Swift Swim
Level: 50
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Protect
- Wave Crash
- Aqua Jet
- Last Respects

Amoonguss @ Sitrus Berry
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 68 Def / 192 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Spore
- Rage Powder
- Pollen Puff
- Clear Smog

Maushold-Four @ Wide Lens
Ability: Technician
Level: 50
Tera Type: Grass
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Population Bomb
- Follow Me
- Taunt
- Protect

Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Grass
EVs: 252 HP / 4 Def / 36 SpA / 196 SpD / 20 Spe
Modest Nature
IVs: 0 Atk
- Electro Shot
- Dragon Pulse
- Flash Cannon
- Body Press
""",
        """
Porygon2 @ Eviolite
Ability: Trace
Level: 50
Shiny: Yes
Tera Type: Ghost
EVs: 252 HP / 140 Def / 36 SpA / 76 SpD / 4 Spe
Modest Nature
IVs: 0 Atk
- Tri Attack
- Shadow Ball
- Recover
- Trick Room

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Fairy
EVs: 252 HP / 236 Atk / 20 SpD
Brave Nature
IVs: 2 Spe
- Facade
- Earthquake
- Headlong Rush
- Protect

Volcarona @ Covert Cloak
Ability: Flame Body
Level: 50
Tera Type: Dragon
EVs: 188 HP / 52 Def / 12 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Overheat
- Struggle Bug
- Rage Powder
- Will-O-Wisp

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
Tera Type: Steel
EVs: 252 HP / 180 Def / 76 SpD
Careful Nature
- Spirit Break
- Reflect
- Light Screen
- Taunt

Annihilape @ Safety Goggles
Ability: Defiant
Level: 50
Tera Type: Fire
EVs: 140 HP / 68 Atk / 12 Def / 60 SpD / 228 Spe
Jolly Nature
- Rage Fist
- Drain Punch
- Protect
- Bulk Up

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Dragon
EVs: 164 HP / 4 Def / 132 SpA / 4 SpD / 204 Spe
Modest Nature
IVs: 0 Atk
- Shadow Ball
- Make It Rain
- Power Gem
- Trick""",
        """
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 172 HP / 116 Atk / 28 Def / 60 SpD / 132 Spe
Adamant Nature
IVs: 18 SpA
- Knock Off
- Flare Blitz
- Parting Shot
- Fake Out

Porygon2 @ Eviolite
Ability: Download
Level: 50
Shiny: Yes
Tera Type: Fighting
EVs: 252 HP / 220 Def / 4 SpA / 28 SpD / 4 Spe
Modest Nature
- Tera Blast
- Ice Beam
- Recover
- Trick Room

Amoonguss @ Mental Herb
Ability: Regenerator
Shiny: Yes
Tera Type: Water
EVs: 236 HP / 164 Def / 108 SpD
Bold Nature
IVs: 21 Atk
- Spore
- Rage Powder
- Pollen Puff
- Sludge Bomb

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Shiny: Yes
Tera Type: Normal
EVs: 140 HP / 180 Atk / 100 Def / 84 SpD / 4 Spe
Brave Nature
IVs: 6 Spe
- Facade
- Headlong Rush
- Earthquake
- Protect

Gholdengo @ Life Orb
Ability: Good as Gold
Shiny: Yes
Tera Type: Dragon
EVs: 44 HP / 4 Def / 196 SpA / 12 SpD / 252 Spe
Timid Nature
IVs: 5 Atk
- Make It Rain
- Shadow Ball
- Nasty Plot
- Protect

Flamigo @ Focus Sash
Ability: Scrappy
Level: 70
Shiny: Yes
Tera Type: Ghost
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
IVs: 0 SpA
- Close Combat
- Brave Bird
- Wide Guard
- Protect""",
        """
Hydrapple @ Leftovers
Ability: Supersweet Syrup
Level: 50
Tera Type: Steel
EVs: 212 HP / 156 SpA / 140 SpD
Quiet Nature
IVs: 0 Atk / 0 Spe
- Syrup Bomb
- Fickle Beam
- Yawn
- Protect

Maushold-Four @ King's Rock
Ability: Technician
Level: 50
Shiny: Yes
Tera Type: Ghost
EVs: 172 HP / 196 Atk / 4 Def / 4 SpD / 132 Spe
Jolly Nature
- Population Bomb
- Taunt
- Follow Me
- Protect

Volcarona @ Safety Goggles
Ability: Flame Body
Level: 50
Shiny: Yes
Tera Type: Dragon
EVs: 252 HP / 68 Def / 36 SpA / 4 SpD / 148 Spe
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
EVs: 204 HP / 156 Atk / 4 Def / 4 SpD / 140 Spe
Adamant Nature
- Wave Crash
- Aqua Jet
- Close Combat
- Protect

Ursaluna-Bloodmoon @ Assault Vest
Ability: Mind's Eye
Level: 50
Tera Type: Normal
EVs: 148 HP / 4 Def / 196 SpA / 4 SpD / 156 Spe
Modest Nature
IVs: 0 Atk
- Hyper Voice
- Blood Moon
- Earth Power
- Vacuum Wave

Gothitelle @ Sitrus Berry
Ability: Shadow Tag
Level: 50
Tera Type: Steel
EVs: 244 HP / 36 Def / 4 SpA / 4 SpD / 220 Spe
Timid Nature
- Psychic
- Helping Hand
- Fake Out
- Trick Room""",
        """
Vivillon @ Focus Sash
Ability: Compound Eyes
Level: 50
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Sleep Powder
- Rage Powder
- Hurricane

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Protect
- Earthquake
- Stomping Tantrum
- Dragon Claw

Primarina @ Sitrus Berry
Ability: Liquid Voice
Level: 50
Tera Type: Poison
EVs: 252 HP / 132 Def / 108 SpA / 4 SpD / 12 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Haze
- Hyper Voice
- Moonblast

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Ghost
EVs: 164 HP / 4 Atk / 36 Def / 68 SpD / 236 Spe
Jolly Nature
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot

Porygon2 @ Eviolite
Ability: Download
Level: 50
Tera Type: Fighting
EVs: 252 HP / 196 SpA / 60 SpD
Quiet Nature
IVs: 0 Spe
- Trick Room
- Recover
- Tera Blast
- Ice Beam

Gholdengo @ Metal Coat
Ability: Good as Gold
Level: 50
Tera Type: Dragon
EVs: 252 HP / 108 Def / 132 SpA / 4 SpD / 12 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Nasty Plot
- Make It Rain
- Shadow Ball""",
        """
Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Ghost
EVs: 244 HP / 4 Atk / 4 Def / 4 SpD / 252 Spe
Jolly Nature
- Knock Off
- Snarl
- Fake Out
- Parting Shot

Amoonguss @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 236 HP / 196 Def / 76 SpD
Calm Nature
IVs: 0 Atk / 26 Spe
- Pollen Puff
- Rage Powder
- Spore
- Protect

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Ghost
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Facade
- Headlong Rush
- Substitute
- Protect

Porygon2 @ Eviolite
Ability: Download
Level: 50
Tera Type: Ghost
EVs: 244 HP / 4 Atk / 36 Def / 92 SpA / 132 SpD
Quiet Nature
IVs: 24 Spe
- Tera Blast
- Ice Beam
- Recover
- Trick Room

Gholdengo @ Metal Coat
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 108 HP / 12 Def / 212 SpA / 4 SpD / 172 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Nasty Plot
- Make It Rain
- Shadow Ball

Primarina @ Mystic Water
Ability: Liquid Voice
Level: 50
Tera Type: Poison
EVs: 244 HP / 4 Def / 188 SpA / 4 SpD / 68 Spe
Modest Nature
IVs: 0 Atk
- Hyper Voice
- Moonblast
- Haze
- Protect""",
        """
Decidueye-Hisui @ Scope Lens
Ability: Scrappy
Level: 50
Tera Type: Flying
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Leaf Blade
- Protect
- Brave Bird
- Triple Arrows

Basculegion (M) @ Mystic Water
Ability: Swift Swim
Level: 50
Tera Type: Water
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
IVs: 10 SpA
- Wave Crash
- Aqua Jet
- Protect
- Last Respects

Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Fairy
EVs: 200 HP / 232 SpA / 76 Spe
Modest Nature
IVs: 24 Atk
- Draco Meteor
- Flash Cannon
- Body Press
- Electro Shot

Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Tailwind
- Weather Ball
- Hurricane
- Protect

Clefairy @ Eviolite
Ability: Friend Guard
Level: 50
Tera Type: Grass
EVs: 252 HP / 252 Def / 4 SpD
Calm Nature
IVs: 0 Atk / 0 Spe
- Follow Me
- Helping Hand
- After You
- Protect

Ursaluna-Bloodmoon @ Life Orb
Ability: Mind's Eye
Level: 50
Tera Type: Normal
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Blood Moon
- Hyper Voice
- Earth Power
- Protect""",
        """
Ribombee (F) @ Focus Sash
Ability: Shield Dust
Level: 50
Shiny: Yes
Tera Type: Dark
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 5 Atk
- Moonblast
- Pollen Puff
- Tailwind
- Fake Tears

Primarina (F) @ Throat Spray
Ability: Liquid Voice
Level: 50
Shiny: Yes
Tera Type: Dragon
EVs: 212 HP / 68 Def / 188 SpA / 4 SpD / 36 Spe
Modest Nature
IVs: 16 Atk
- Protect
- Hyper Voice
- Moonblast
- Perish Song

Gothitelle (F) @ Mental Herb
Ability: Shadow Tag
Level: 50
Shiny: Yes
Tera Type: Water
EVs: 244 HP / 92 Def / 4 SpA / 156 SpD / 12 Spe
Calm Nature
IVs: 14 Atk
- Protect
- Fake Out
- Psychic
- Trick Room

Ursaluna-Bloodmoon @ Life Orb
Ability: Mind's Eye
Level: 50
Tera Type: Normal
EVs: 4 HP / 4 Def / 252 SpA / 108 SpD / 140 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Hyper Voice
- Earth Power
- Blood Moon

Incineroar (M) @ Sitrus Berry
Ability: Intimidate
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 204 HP / 252 Atk / 4 Def / 4 SpD / 44 Spe
Adamant Nature
- Protect
- Fake Out
- Knock Off
- Flare Blitz

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Shiny: Yes
Tera Type: Steel
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Make It Rain
- Shadow Ball
- Thunderbolt
- Power Gem""",
        """
Whimsicott @ Covert Cloak
Ability: Prankster
Level: 50
Tera Type: Fire
EVs: 252 HP / 36 Def / 4 SpA / 212 SpD / 4 Spe
Bold Nature
IVs: 0 Atk
- Tailwind
- Moonblast
- Sunny Day
- Encore

Ursaluna-Bloodmoon @ Life Orb
Ability: Mind's Eye
Level: 50
Tera Type: Normal
EVs: 4 HP / 4 Def / 212 SpA / 60 SpD / 228 Spe
Modest Nature
IVs: 0 Atk
- Earth Power
- Blood Moon
- Hyper Voice
- Protect

Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Grass
EVs: 252 HP / 76 Def / 68 SpA / 4 SpD / 108 Spe
Modest Nature
IVs: 0 Atk
- Electro Shot
- Flash Cannon
- Body Press
- Draco Meteor

Politoed @ Sitrus Berry
Ability: Drizzle
Level: 50
Shiny: Yes
Tera Type: Grass
EVs: 252 HP / 164 Def / 12 SpA / 76 SpD / 4 Spe
Bold Nature
IVs: 0 Atk
- Perish Song
- Haze
- Weather Ball
- Protect

Typhlosion-Hisui @ Choice Specs
Ability: Blaze
Level: 50
Shiny: Yes
Tera Type: Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Eruption
- Heat Wave
- Infernal Parade
- Solar Beam

Indeedee-F @ Psychic Seed
Ability: Psychic Surge
Level: 50
Tera Type: Fairy
EVs: 252 HP / 244 Def / 4 SpA / 4 SpD / 4 Spe
Bold Nature
IVs: 0 Atk
- Trick Room
- Follow Me
- Imprison
- Psychic""",
        """
Hydreigon @ Choice Specs
Ability: Levitate
Level: 50
Tera Type: Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Dark Pulse
- Heat Wave
- Snarl

Indeedee-F @ Safety Goggles
Ability: Psychic Surge
Level: 50
Tera Type: Ghost
EVs: 252 HP / 132 Def / 4 SpA / 100 SpD / 20 Spe
Bold Nature
IVs: 0 Atk
- Hyper Voice
- Imprison
- Trick Room
- Follow Me

Gholdengo @ Life Orb
Ability: Good as Gold
Level: 86
Tera Type: Dragon
EVs: 52 HP / 4 Def / 212 SpA / 60 SpD / 180 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Protect
- Nasty Plot

Murkrow @ Eviolite
Ability: Prankster
Level: 52
Tera Type: Ghost
EVs: 252 HP / 220 SpD / 36 Spe
Calm Nature
IVs: 0 Atk
- Foul Play
- Protect
- Haze
- Tailwind

Sneasler @ Psychic Seed
Ability: Unburden
Level: 50
Tera Type: Dark
EVs: 236 HP / 252 Atk / 4 Def / 4 SpD / 12 Spe
Adamant Nature
- Close Combat
- Dire Claw
- Throat Chop
- Swords Dance

Basculegion @ Choice Scarf
Ability: Adaptability
Level: 50
Tera Type: Grass
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Wave Crash
- Last Respects
- Tera Blast
- Aqua Jet""",
        """
Garchomp @ Clear Amulet
Ability: Sand Veil
Level: 50
Tera Type: Ground
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Protect
- Earthquake
- Dragon Claw
- Swords Dance

Talonflame (F) @ Covert Cloak
Ability: Gale Wings
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Brave Bird
- Tailwind
- Will-O-Wisp
- Taunt

Tyranitar (M) @ Assault Vest
Ability: Sand Stream
Level: 50
Tera Type: Flying
EVs: 252 HP / 244 Atk / 12 SpD
Adamant Nature
- Rock Slide
- Assurance
- Stone Edge
- Low Kick

Gholdengo @ Choice Specs
Ability: Good as Gold
Level: 50
Tera Type: Steel
EVs: 252 HP / 228 SpA / 28 Spe
Modest Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Thunderbolt
- Trick

Brambleghast (F) @ Focus Sash
Ability: Wind Rider
Level: 50
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Poltergeist
- Power Whip
- Strength Sap
- Shadow Sneak

Glimmora (F) @ Power Herb
Ability: Toxic Debris
Level: 50
Tera Type: Grass
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Meteor Beam
- Spiky Shield
- Earth Power
- Sludge Bomb""",
        """
Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Tera Type: Dark
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Swords Dance
- Protect

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
Tera Type: Fire
EVs: 252 HP / 116 Atk / 4 Def / 124 SpD / 12 Spe
Adamant Nature
- Wood Hammer
- Grassy Glide
- U-turn
- Fake Out

Clefable @ Safety Goggles
Ability: Unaware
Level: 50
Tera Type: Steel
EVs: 252 HP / 212 Def / 4 SpA / 36 SpD / 4 Spe
Bold Nature
IVs: 0 Atk
- Follow Me
- Helping Hand
- Moonblast
- Protect

Volcarona @ Leftovers
Ability: Flame Body
Level: 50
Tera Type: Dragon
EVs: 252 HP / 132 Def / 36 SpA / 4 SpD / 84 Spe
Modest Nature
IVs: 0 Atk
- Heat Wave
- Giga Drain
- Quiver Dance
- Protect

Sneasler (F) @ Grassy Seed
Ability: Unburden
Level: 50
Tera Type: Stellar
EVs: 164 HP / 76 Atk / 12 Def / 4 SpD / 252 Spe
Adamant Nature
- Protect
- Fake Out
- Close Combat
- Dire Claw

Dondozo @ Covert Cloak
Ability: Unaware
Level: 50
Tera Type: Steel
EVs: 244 HP / 12 Atk / 4 Def / 244 SpD / 4 Spe
Careful Nature
- Liquidation
- Fissure
- Yawn
- Curse""",
        """
Indeedee-F @ Psychic Seed
Ability: Psychic Surge
Level: 50
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Follow Me
- Helping Hand
- Psychic
- Trick Room

Hatterene @ Life Orb
Ability: Magic Bounce
Level: 50
Tera Type: Fire
EVs: 212 HP / 212 SpA / 84 SpD
Quiet Nature
IVs: 0 Atk / 0 Spe
- Expanding Force
- Dazzling Gleam
- Trick Room
- Mystical Fire

Torkoal @ Choice Specs
Ability: Drought
Level: 50
Tera Type: Fire
EVs: 252 HP / 252 SpA / 4 SpD
Quiet Nature
IVs: 0 Atk / 0 Spe
- Eruption
- Heat Wave
- Earth Power
- Weather Ball

Araquanid @ Clear Amulet
Ability: Water Bubble
Level: 50
Tera Type: Grass
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Liquidation
- Leech Life
- Wide Guard
- Protect

Gallade @ Scope Lens
Ability: Sharpness
Level: 50
Tera Type: Grass
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Wide Guard
- Sacred Sword
- Psycho Cut
- Trick Room

Lilligant-Hisui @ Focus Sash
Ability: Chlorophyll
Level: 50
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- After You
- Triple Axel
- Close Combat
- Sleep Powder""",
        """
Lokix @ Safety Goggles
Ability: Tinted Lens
Level: 50
Tera Type: Stellar
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- X-Scissor
- Swords Dance
- Throat Chop
- Protect

Venonat @ Eviolite
Ability: Compound Eyes
Level: 50
Tera Type: Grass
EVs: 252 HP / 164 Def / 92 SpD
Bold Nature
IVs: 0 Atk / 0 Spe
- Sleep Powder
- Rage Powder
- Toxic
- Morning Sun

Araquanid @ Mystic Water
Ability: Water Bubble
Level: 50
Tera Type: Water
EVs: 252 HP / 236 Atk / 20 Def
Adamant Nature
- Liquidation
- Leech Life
- Wide Guard
- Protect

Ribombee @ Focus Sash
Ability: Shield Dust
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Sunny Day
- Tailwind
- Moonblast
- Fake Tears

Volcarona @ Choice Specs
Ability: Flame Body
Level: 50
Tera Type: Fire
EVs: 20 HP / 252 SpA / 236 Spe
Modest Nature
IVs: 0 Atk
- Fiery Dance
- Heat Wave
- Overheat
- Bug Buzz

Kleavor @ Assault Vest
Ability: Sharpness
Level: 50
Tera Type: Water
EVs: 68 HP / 252 Atk / 188 Spe
Adamant Nature
- Stone Axe
- X-Scissor
- Rock Blast
- Aerial Ace""",
        """
Sneasler @ Psychic Seed
Ability: Unburden
Level: 50
Tera Type: Dark
EVs: 4 HP / 236 Atk / 28 Def / 4 SpD / 236 Spe
Adamant Nature
- Protect
- Close Combat
- Dire Claw
- Throat Chop

Kleavor @ Focus Sash
Ability: Sharpness
Level: 50
Tera Type: Stellar
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Protect
- Stone Axe
- Night Slash
- Close Combat

Typhlosion-Hisui @ Charcoal
Ability: Frisk
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Eruption
- Shadow Ball
- Heat Wave

Hydreigon @ Razor Claw
Ability: Levitate
Level: 50
Tera Type: Fire
EVs: 4 HP / 4 Def / 244 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Draco Meteor
- Heat Wave
- Focus Energy

Indeedee-F @ Rocky Helmet
Ability: Psychic Surge
Level: 50
Tera Type: Grass
EVs: 252 HP / 220 Def / 36 SpD
Calm Nature
IVs: 0 Atk / 27 Spe
- Helping Hand
- Follow Me
- Trick Room
- Imprison

Whimsicott @ Covert Cloak
Ability: Prankster
Level: 50
Tera Type: Dark
EVs: 252 HP / 4 Def / 4 SpA / 116 SpD / 132 Spe
Timid Nature
IVs: 0 Atk
- Moonblast
- Encore
- Tailwind
- Sunny Day""",
        """
Archaludon @ Assault Vest
Ability: Stamina
Level: 50
Tera Type: Grass
EVs: 212 HP / 4 Def / 36 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Draco Meteor
- Electro Shot
- Flash Cannon
- Body Press

Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hurricane
- Weather Ball
- Helping Hand
- Tailwind

Basculegion @ Life Orb
Ability: Swift Swim
Level: 50
Tera Type: Grass
EVs: 220 Atk / 36 Def / 252 Spe
Adamant Nature
- Protect
- Wave Crash
- Aqua Jet
- Last Respects

Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 50
Tera Type: Dark
EVs: 252 HP / 100 Def / 156 SpD
Calm Nature
IVs: 0 Atk / 27 Spe
- Spore
- Pollen Puff
- Rage Powder
- Clear Smog

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Tera Type: Dark
EVs: 252 HP / 116 Atk / 4 Def / 20 SpD / 116 Spe
Adamant Nature
- Protect
- Kowtow Cleave
- Sucker Punch
- Swords Dance

Ludicolo @ Mystic Water
Ability: Swift Swim
Level: 50
Tera Type: Water
EVs: 12 HP / 36 Def / 236 SpA / 4 SpD / 220 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Muddy Water
- Giga Drain
- Ice Beam""",
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
