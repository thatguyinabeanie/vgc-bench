# VGC-Bench

A benchmark for training AI agents to play competitive Pokémon (Video Game Championships format) using both supervised and reinforcement learning approaches.

📄 **Paper**: [VGC-Bench: A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon](https://arxiv.org/abs/2506.10326)

## Features

- **Supervised Learning (SL)**: Learn from human VGC battle logs with open team sheets
- **Reinforcement Learning (RL)**: Fine-tune agents using PPO with 3 PSRO methods (selfplay, pfsp, p2sro)
- **Cross-play Evaluation**: ELO rating system for comparing different agents
- **VGC Format Support**: Doubles battles, bring 6 pick 4, with full game mechanics

## Quick Start

### Prerequisites

- Python 3.10-3.12
- Node.js and npm
- [mise](https://mise.jdx.dev/) (recommended) or manual setup

### Option 1: Using mise (Recommended)

```bash
# Install mise if you haven't already
# See: https://mise.jdx.dev/getting-started.html

# Run the quickstart
mise run quickstart

# Start the Pokemon Showdown server (keep this running)
mise run server

# Train an agent
mise run train      # RL training
mise run pretrain   # Supervised learning
```

### Option 2: Manual Setup

```bash
# 1. Clone submodules
git submodule update --init --recursive

# 2. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# 3. Initial Pokemon Showdown setup
cd pokemon-showdown
node pokemon-showdown start --no-security
# Wait for "Test your server at http://localhost:8000" then Ctrl+C
cd ..

# 4. Download game data
python vgc_bench/scrape_data.py
```

## Usage

### Training

Always start the Pokemon Showdown server first:

```bash
# In a separate terminal
cd pokemon-showdown && node pokemon-showdown start --no-security
```

Then run training:

```bash
# Reinforcement Learning with PPO
python vgc_bench/train.py
# or use the script: ./train.sh

# Supervised Learning from battle logs
python vgc_bench/pretrain.py
# or use the script: ./pretrain.sh

# Evaluate agents and calculate ELO ratings
python vgc_bench/eval.py
# or use the script: ./eval.sh

# Play on online servers
python vgc_bench/play.py --help
```

### Available mise Tasks

```bash
mise tasks         # List all available tasks
mise run fmt       # Format code with black and isort
mise run lint      # Run linting checks
mise run test      # Run tests
mise run clean     # Clean up generated files
```

## Project Structure

```
vgc_bench/
├── agent.py          # Agent implementation for battles
├── env.py            # Gymnasium environment wrapper
├── policy.py         # Masked actor-critic policy
├── train.py          # RL training (PPO + PSRO methods)
├── pretrain.py       # Supervised learning pipeline
├── eval.py           # Cross-play evaluation
├── play.py           # Online play interface
├── scrape_logs.py    # Battle log scraper
├── logs2trajs.py     # Convert logs to trajectories
└── scrape_data.py    # Download game data (moves, abilities, items)
```

## Configuration

Training parameters can be configured via command-line arguments:

```bash
python vgc_bench/train.py --help
python vgc_bench/pretrain.py --help
```

Or modify the shell scripts for common configurations:

- `train.sh` - RL training presets
- `pretrain.sh` - SL training presets
- `eval.sh` - Evaluation presets

## Data

Pre-collected Gen 9 VGC battle logs with open team sheets:
🤗 [VGC Battle Logs](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs)

## Development

```bash
# Format code
mise run fmt

# Check code quality
mise run lint
mise run typecheck

# Run tests
mise run test
```

## Technical Details

- **Environment**: Text-based observations converted to embeddings using sentence-transformers
- **Action Space**: Discrete(10) - up to 4 moves + 6 switches with action masking
- **PSRO Methods**:
  - `selfplay`: Train against copies of itself
  - `pfsp`: Prioritized fictitious self-play
  - `p2sro`: Policy-space response oracles
- **Team Format**: VGC doubles (bring 6, pick 4)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vgc-bench2024,
  title={VGC-Bench: A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon},
  author={[Authors]},
  journal={arXiv preprint arXiv:2506.10326},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
