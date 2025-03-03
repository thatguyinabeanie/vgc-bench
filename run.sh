#!/bin/bash
if [[ $PATH != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi

teams=(1 3 10 30)
ports=(8000 8001 8002 8003)
devices=("cuda:0" "cuda:1" "cuda:2" "cuda:3")
for i in "${!teams[@]}"; do
    num_teams=${teams[$i]}
    port=${ports[$i]}
    device=${devices[$i]}
    (
        cd pokemon-showdown
        node pokemon-showdown start "$port" --no-security > /dev/null 2>&1
    ) &
    (
        python dexter/pretrain.py --num_teams "$num_teams" --device "$device" 2> debug"$port".log
        python dexter/train.py --num_teams "$num_teams" --port "$port" --device "$device" > debug"$port".log 2>&1
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Run $i died"
            break
        fi
    ) &
done
wait
