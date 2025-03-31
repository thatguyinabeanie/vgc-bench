#!/bin/bash

if [[ $PATH != "/scratch/cluster/cangliss/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/bin:$PATH"
fi

num_teams=1
filepath=../UT-masters-thesis3/results/saves-fp/"$num_teams"-teams/589824.zip
port=8000

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown-"$port"
        node pokemon-showdown start "$port" --no-security > /dev/null 2>&1 &
        echo $!
    )
}

echo "Starting Showdown server for pretraining process..."
showdown_pid=$(start_showdown "$port")
echo "Starting evaluation..."
python dexter/eval.py --filepath "$filepath" --num_teams "$num_teams" --port "$port"
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Evaluation process died with exit status $exit_status"
fi
kill $showdown_pid
