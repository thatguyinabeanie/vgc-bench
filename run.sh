#!/bin/bash

if [[ $PATH != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi

teams=(1 3 10 30)
ports=(8000 8001 8002 8003)
devices=("cuda:0" "cuda:1" "cuda:2" "cuda:3")

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown-"$port"
        node pokemon-showdown start "$port" --no-security > /dev/null 2>&1 &
        echo $!
    )
}

start_training() {
    local i=$1
    local num_teams=${teams[$i]}
    local port=${ports[$i]}
    local device=${devices[$i]}

    python dexter/pretrain.py --num_teams "$num_teams" --device "$device" 2> debug"$port".log

    while true; do
        echo "Starting Showdown server for training process $i..."
        showdown_pid=$(start_showdown "$port")
        showdown_pids[$i]="$showdown_pid"
        echo "Starting training process $i..."
        timeout 3600 python dexter/train.py --num_teams "$num_teams" --port "$port" --device "$device" > debug"$port".log 2>&1 &
        train_pids[$i]=$!
        wait "${train_pids[$i]}"
        kill -9 "$showdown_pid" 2>/dev/null
    done
}

# Run training processes in parallel
for i in "${!teams[@]}"; do
    start_training "$i" &
done

wait
