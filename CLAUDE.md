# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

VGC-Bench is a benchmark for competitive Pokémon (Video Game Championships)
that implements both supervised learning (SL) and reinforcement learning (RL)
approaches for training AI agents.
The project focuses on generalizing across diverse team strategies in
competitive Pokémon battles.

### Development Setup

Initial setup (requires Node.js/npm and Python 3.10-3.12)

```bash
git submodule update --init --recursive

# Run until server starts, then Ctrl+C
cd pokemon-showdown && node pokemon-showdown start --no-security
cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
python vgc_bench/scrape_data.py  # Download game data
```

### Running Code

```bash
# Always start Pokemon Showdown server first in a separate terminal:
cd pokemon-showdown && node pokemon-showdown start --no-security

# Main training commands:
python vgc_bench/train.py        # RL training with PPO/PSRO methods
python vgc_bench/pretrain.py     # Supervised learning from battle logs
python vgc_bench/eval.py         # Cross-play evaluation and ELO ratings
python vgc_bench/play.py         # Play on online servers

# Utility scripts:
./train.sh      # Streamlined RL training
./pretrain.sh   # Streamlined SL training
./eval.sh       # Streamlined evaluation
```

### Code Quality

```bash
# Format code
black vgc_bench
isort vgc_bench

# Lint and typecheck
black --check vgc_bench
isort --check-only vgc_bench
pyright vgc_bench

# Run tests
pytest
```

## Architecture

### Core Training Components

- **vgc_bench/agent.py**: Agent implementation for playing battles
- **vgc_bench/env.py**: Gymnasium-based Pokémon Showdown environment wrapper
- **vgc_bench/policy.py**: Masked actor-critic policy for handling legal actions
- **vgc_bench/train.py**
  - RL training with PPO and 3 PSRO variants (selfplay, pfsp, p2sro)
- **vgc_bench/pretrain.py**: Behavioral cloning from VGC battle logs

### Data Processing

- **vgc_bench/scrape_logs.py**: Scrapes battle logs from Pokémon Showdown
- **vgc_bench/logs2trajs.py**: Converts logs to state-action trajectories
- **vgc_bench/scrape_data.py**: Downloads moves, abilities, items data

### Key Dependencies

- **poke-env**: Custom fork for Pokémon battle environment
- **stable-baselines3**: PPO implementation
- **imitation**: Behavioral cloning
- **gymnasium/supersuit**: Environment wrappers
- **sentence-transformers**: Text embeddings for observations

### Environment Details

The environment uses:

- Observation space: Dict with multiple components (battle state, teams, legal actions)
- Action space: Discrete(10) - up to 4 moves + 6 switches
- Masked actions via `action_masks()` method
- Text-based observations converted to embeddings

## Important Notes

1. **Pokemon Showdown Server**: Must be running before any training/evaluation
2. **Team Format**: Uses VGC format (doubles, bring 6 pick 4)
3. **Observation Processing**: Text observations are embedded using sentence-transfor:wrappers:mers
4. **Action Masking**: Critical for training - only legal actions are allowed
5. **PSRO Methods**: Supports self play, prioritized fictitious self-play, p2sro
6. **Evaluation**: Cross-play between agents calculates ELO ratings

## Common Issues

- If `mise install` fails with path separator error, manually run the setup commands
- The `imitation` package has dependency conflicts with gymnasium>=1.0.0.
  - It's only needed for behavioral cloning (pretrain.py).
  - if you need to use pretrain functionality,install it separately
    - `pip install "gymnasium~=0.29.1" "imitation>=1.0.0"`
- Ensure pokemon-showdown submodule is properly initialized
- Python version should be 3.10-3.12 for compatibility
- Always check that the Showdown server is running before training/evaluation
