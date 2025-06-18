# VGC-Bench MCP Integration Design

This document outlines how to create an MCP (Model Context Protocol) server for VGC-Bench, enabling any LLM to interface with the Pokémon battle system.

## Overview

The VGC-Bench MCP Server provides a standardized interface for LLMs to:
- Play Pokémon battles with full game state awareness
- Analyze battle situations and suggest optimal moves
- Build competitive teams with strategic insights
- Review battle replays and provide commentary
- Monitor and optimize RL training processes

## MCP Server Architecture

### Tools

#### Battle Actions
- `make_move(move_id: int, target?: int)` - Execute a move in battle
  - `move_id`: Index of the move to use (0-3)
  - `target`: Optional target for moves in doubles (0-3)
  - Returns: Move result and updated battle state

- `switch_pokemon(slot: int)` - Switch to a different Pokémon
  - `slot`: Team position to switch to (0-5)
  - Returns: Switch result and updated battle state

#### Analysis Tools
- `analyze_battle_state()` - Get detailed analysis of current situation
  - Returns: Strategic analysis including type matchups, speed tiers, threat assessment

- `calculate_damage(move: str, attacker_slot: int, defender_slot: int)` - Damage calculations
  - Returns: Min/max damage, KO probability, after-effects

- `suggest_actions()` - Get AI recommendations with explanations
  - Returns: Ranked list of actions with strategic reasoning

#### Team Building
- `analyze_team(team: List[Dict])` - Evaluate team composition
  - Returns: Synergy analysis, coverage gaps, meta matchups

- `suggest_team_member(current_team: List[Dict], role: str)` - Recommend Pokémon
  - Returns: Pokémon suggestions with movesets and EV spreads

### Resources

#### Battle State
- `/battle/{id}/state` - Current battle state
  - Teams with HP, status, stat changes
  - Field conditions (weather, terrain)
  - Turn count and game phase

- `/battle/{id}/history` - Move history and turn log
  - Chronological list of all actions
  - Damage dealt, effects triggered

- `/battle/{id}/legal_actions` - Available moves this turn
  - Valid moves with targets
  - Switch options

#### Game Data
- `/pokemon/{name}` - Pokédex data
  - Base stats, types, abilities
  - Learnable moves, evolution chain

- `/moves/{name}` - Move information
  - Power, accuracy, effects
  - Type and category

- `/abilities/{name}` - Ability details
  - Effect description
  - Interaction notes

- `/items/{name}` - Item information
  - Effect and activation conditions
  - Usage statistics

#### Meta Information
- `/teams/meta` - Popular team compositions
  - Usage statistics
  - Common cores and partnerships

- `/replays/{id}` - Historical battle data
  - Full battle logs for analysis
  - Filterable by rating, team, date

### Prompts

- `battle-assistant` - Interactive battle helper
  - Provides move suggestions with explanations
  - Answers rules questions
  - Calculates probabilities

- `team-builder` - Team construction assistant
  - Guides through team building process
  - Suggests synergistic combinations
  - Optimizes for specific strategies

- `replay-analyst` - Post-battle analysis
  - Identifies key turning points
  - Suggests alternative lines
  - Rates play quality

- `training-coach` - RL training optimization
  - Monitors agent performance
  - Suggests curriculum adjustments
  - Identifies learning plateaus

### Subscriptions

- `battle-updates` - Real-time battle state changes
  - Streams battle events as they occur
  - Includes state diffs for efficiency

- `training-metrics` - Live training progress
  - Win rates, ELO progression
  - Policy entropy, value loss
  - Behavioral patterns

## Implementation Details

### MCP Server Wrapper
```python
# vgc_bench/mcp_server.py
class VGCBenchMCPServer:
    def __init__(self):
        self.battles = {}  # Active battle instances
        self.env_wrapper = VGCEnvironment()
        
    async def handle_make_move(self, battle_id: str, move_id: int, target: Optional[int]):
        battle = self.battles[battle_id]
        action = self.encode_action(move_id, target)
        obs, reward, done, info = battle.step(action)
        return self.format_battle_update(obs, reward, done, info)
```

### Stateful Battle Management
- Each LLM client receives a unique battle session ID
- Server maintains battle state between requests
- Supports multiple concurrent battles
- Handles disconnections gracefully

### Rich Context Provision
- Convert game state to natural language descriptions
- Include strategic context:
  - Type effectiveness calculations
  - Speed order predictions  
  - Common move patterns
  - Win condition analysis
- Provide uncertainty estimates for hidden information

### Integration Points

1. **Direct Agent Mode**: LLM makes all decisions
2. **Advisory Mode**: LLM assists human player
3. **Analysis Mode**: LLM reviews completed battles
4. **Training Mode**: LLM guides RL optimization

## Usage Example

```python
# Client code (works with any MCP-compatible LLM)
async with mcp.Client("vgc-bench-server") as client:
    # Start a new battle
    battle = await client.call("start_battle", team=my_team)
    
    # Main battle loop
    while not battle.done:
        # Get current state
        state = await client.resource(f"/battle/{battle.id}/state")
        
        # Get LLM's decision
        action = await llm.decide(state, client.tools)
        
        # Execute the action
        result = await client.call(action.tool, **action.params)
```

## Benefits

1. **Universal LLM Support**: Any MCP-compatible LLM can play Pokémon
2. **Standardized Interface**: Consistent API across different battle formats
3. **Rich Context**: LLMs receive comprehensive game state information
4. **Extensible**: Easy to add new analysis tools and resources
5. **Educational**: Great for learning competitive Pokémon strategies

## Next Steps

1. Implement core MCP server with basic battle tools
2. Add comprehensive game state serialization
3. Create example clients for popular LLMs
4. Build evaluation suite comparing different LLMs
5. Develop specialized prompts for different play styles