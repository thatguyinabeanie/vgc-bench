#!/bin/bash
[[ ":$PATH:" != *"/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]] && export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"

# Configuration variables
NUM_TEAMS=16
PORT=8000
DEVICE="cuda:3"
LOG_FILE="debug.log"

{
    python dexter/pretrain.py --num_teams "$NUM_TEAMS" --port "$PORT" --device "$DEVICE"
    # while true; do
    #     python dexter/train.py --num_teams "$NUM_TEAMS" --port "$PORT" --device "$DEVICE"
    #     [[ $? -ne 0 ]] && sleep 10
    # done
} > "$LOG_FILE" 2>&1
