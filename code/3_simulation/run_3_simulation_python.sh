#!/bin/bash

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp

echo "Running Python"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"

set -e

echo "Generating Python Binaries"
python3 "python/generate_binaries.py" 152 True

echo "Executing Python Simulation"
python3 "python/gamma_exec.py" 152

echo "Calculating Welfare"
echo "$script_dir"
cd "$script_dir"
python3 "python/compute_baseline_welfares.py" 152 True

echo "Calculating Consumer Surpluses"
echo "$script_dir"
cd "$script_dir"
python3 "python/consumer_surplus.py" config.json 152 True

echo "Generating plots of robustness"
echo "$script_dir"
cd "$script_dir"
python3 "python/gen_stability_csv.py" 152 True

cd "$script_dir"
rm -rf python/Checkpoints/*
rm -rf python/runs/*.csv
rm -rf python/runs/*.png

