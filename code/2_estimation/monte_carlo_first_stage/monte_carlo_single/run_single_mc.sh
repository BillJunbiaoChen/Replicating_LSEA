#!/bin/bash

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp
set -e

# amenity_supply/* b
echo "Running single trial of the monte carlo data generation"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"

# Create necessary directories
mkdir -p Data Results

echo "Starting Monte Carlo simulation..."

# Step 1: Generate data
echo "Generating data..."
cd code
#julia generate_mc.jl

# Step 2: Run MNL estimation
cd "$script_dir"
cd code
echo "Running MNL estimation..."
#stata-mp -b do mnl.do

# Step 3: Run final estimation
cd "$script_dir"
cd code
echo "Running final estimation..."
julia estimate_mc.jl

echo "Monte Carlo simulation completed."
echo "Results are stored in the Results directory."