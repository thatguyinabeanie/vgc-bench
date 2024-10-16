#!/bin/bash
if [[ ":$PATH:" != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi
python src/train.py --teams 0 --opp_teams 0 1 2 3 --port 8000 --device cuda:0 &
python src/train.py --teams 0 --opp_teams 0 1 2 3 4 5 6 7 --port 8001 --device cuda:1 &
python src/train.py --teams 0 --opp_teams 0 1 2 3 4 5 6 7 8 9 10 11 --port 8002 --device cuda:2 &
python src/train.py --teams 0 --opp_teams 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --port 8003 --device cuda:3 &
wait
