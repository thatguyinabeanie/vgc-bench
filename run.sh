#!/bin/bash
if [[ ":$PATH:" != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi
python src/train.py --teams 0 --opp_teams 1 --port 8000 --device cuda:0 &
python src/train.py --teams 0 --opp_teams 2 --port 8001 --device cuda:1 &
python src/train.py --teams 0 --opp_teams 3 --port 8002 --device cuda:2 &
python src/train.py --teams 0 --opp_teams 4 --port 8003 --device cuda:3 &
wait
