#!/bin/bash

if [[ $PATH != "/scratch/cluster/cangliss/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/bin:$PATH"
fi

teams=(1 2 5 10)
ports=(8000 8001 8002 8003)
devices=("cuda:0" "cuda:1" "cuda:2" "cuda:3")

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown
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
    python dexter/train.py --num_teams "$num_teams" --port "$port" --device "$device" --behavior_clone --fictitious_play > debug"$port".log 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Training process $i died with exit status $exit_status"
    fi
    kill $showdown_pid
}

while true; do
    for i in "${!teams[@]}"; do
        start_training "$i" &
        sleep 1
    done
    wait
    sleep 10
done
