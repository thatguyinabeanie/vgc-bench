#!/bin/bash
if [[ ":$PATH:" != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi
python src/run.py --num_teams 16 --port 8000 --device cuda:3
