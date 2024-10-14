#!/bin/bash
ports=(8001 8002 8003 8004)
for port in "${ports[@]}"
do
    pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid on port $port"
        kill -9 $pid
    else
        echo "No process running on port $port"
  fi
done
