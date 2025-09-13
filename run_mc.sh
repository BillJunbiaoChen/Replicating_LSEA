#!/bin/bash


set -e

#NUM_CORES=$(sysctl -n hw.ncpu)
export JULIA_NUM_THREADS=18

echo "Julia threading usage set to use number of core: $JULIA_NUM_THREADS"

# Run setup scripts
julia setup.jl
Rscript setup.R

echo "Creating Virtual Environment for Executing Python Simulation"

pip3 install --upgrade pip
pip3 install virtualenv
virtualenv venv
source ./venv/bin/activate

echo "Installing Python Libraries for Executing Python Simulation"
echo "$PWD"
pip3 install -r requirements.txt

# Change permissions of all files in the current directory
 sudo chmod +x ./*

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp

echo "Running code/2_estimation/monte_carlo_first_stage..."
master_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$master_script_dir"
echo "$master_script_dir"
source "code/2_estimation/monte_carlo_first_stage/code/run_all.sh"
echo "Finished code/monte_carlo_first_stage..."