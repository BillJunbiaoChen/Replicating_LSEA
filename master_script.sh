#!/bin/bash

set -e

#NUM_CORES=$(sysctl -n hw.ncpu)
export JULIA_NUM_THREADS=7

echo "Julia threading usage set to use number of core: $JULIA_NUM_THREADS"

# Run setup scripts
julia setup.jl
Rscript setup.R


echo "Creating Virtual Environment for Executing Python Simulation"


pip3 install --upgrade pip
python3 -m venv venv
source ./venv/bin/activate

echo "Installing Python Libraries for Executing Python Simulation"
echo "$PWD"
pip3 install -r requirements.txt

# Change permissions of all files in the current directory
sudo chmod +x ./*

# Master script directory setup
echo "Running code/0_dataprep..."
master_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$master_script_dir"
echo "$master_script_dir"
source "code/0_dataprep/run_0_dataprep.sh"
echo "Finished code/0_dataprep..."

echo "Running code/1_descriptives..."
cd "$master_script_dir"
echo "$master_script_dir"
source "code/1_descriptives/run_1_descriptives.sh"
echo "Finished code/1_descriptives..."

echo "Running code/2_estimation..."
cd "$master_script_dir"
echo "$master_script_dir"
source "code/2_estimation/run_2_estimation.sh"
echo "Finished code/2_estimation..."

echo "Running code/3_simulation python version..."
cd "$master_script_dir"
echo "$master_script_dir"
source "code/3_simulation/run_3_simulation_python.sh"
echo "Finished code/3_simulation python version..."

#echo "Running code/3_simulation julia version..."
#cd "$master_script_dir"
#echo "$master_script_dir"
#source "code/3_simulation/run_3_simulation_julia.sh"
#echo "Finished code/3_simulation julia version..."

echo "Running code/3_simulation plots..."
cd "$master_script_dir"
echo "$master_script_dir"
source "code/3_simulation/run_3_simulation_plots.sh"
echo "Finished code/3_simulation plots..."
