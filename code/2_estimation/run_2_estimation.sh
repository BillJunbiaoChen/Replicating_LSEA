#!/bin/bash

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp
set -e

# amenity_supply/* b
echo "Running amenity_supply/estimation_amenity_bootstrap_bayesian.jl"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"
julia amenity_supply/estimation_amenity_bootstrap_bayesian.jl

# demand/*
echo "Running demand/master_demand.do"
echo "$script_dir"
cd "$script_dir"
stata-mp -e do "demand/master_demand.do"

echo "Running demand/construct_wtp.py"
echo "$script_dir"
cd "$script_dir"
python3 "demand/construct_wtp.py"

# housing_supply/*
echo "Running housing_supply/1_housing_supply.R"
echo "$script_dir"
cd "$script_dir"
Rscript "housing_supply/1_housing_supply.R"

echo "Finished running 2_estimation. To run Monte Carlo First Stage, please execute mc_run.sh in root directory"
