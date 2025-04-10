#!/bin/bash

if [[ $PATH != "/scratch/cluster/cangliss/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/bin:$PATH"
fi

port=8004

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown
        node pokemon-showdown start "$port" --no-security > /dev/null 2>&1 &
        echo $!
    )
}

echo "Starting Showdown server for pretraining process..."
showdown_pid=$(start_showdown "$port")
echo "Starting evaluation..."
python dexter/eval.py --port "$port"
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Evaluation process died with exit status $exit_status"
fi
kill $showdown_pid
