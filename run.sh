#!/bin/bash
if [[ ":$PATH:" != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi
for i in {0..3}; do
    python src/train.py --run_id $i &
done
wait
