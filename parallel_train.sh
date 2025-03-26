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

    echo "Starting Showdown server for training process $i..."
    showdown_pid=$(start_showdown "$port")
    echo "Starting training process $i..."
    python dexter/train.py --num_teams "$num_teams" --port "$port" --device "$device" --behavior_clone --double_oracle > debug"$port".log 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Training process $i died with exit status $exit_status"
    fi
    kill $showdown_pid
}

child_pids=()
# trap 'echo "Terminating all processes..."; kill ${child_pids[@]}' SIGINT SIGTERM
for i in "${!teams[@]}"; do
    # while true; do
    start_training "$i" &
    #     sleep 10
    # done &
    # child_pids+=($!)
    sleep 1
done
wait
