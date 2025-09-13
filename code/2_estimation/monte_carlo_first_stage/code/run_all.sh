#!/bin/bash

set -e

echo "Generating Data Now"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Navigate to the directory containing data_gen files
cd ./Generate_data/

export JULIA_NUM_THREADS=2

# Run Julia Files to Generate Data
# Loop through all .jl files containing "gen_"
#for file in *.jl; do
#  if [[ $file =~ gen_ ]]; then
    # Run the file using Julia interpreter
#    julia "$file"
#  fi
#done

# Go back to the original directory
cd ../

echo "Fitting MNL Model to Choice Data"
export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp
#stata-mp -e do "MNL.do" 

echo "Estimating Parameters via 3 Models"

# Navigate to the directory for estimation
cd ./Main_functions/

# Loop through all .jl files containing "Estimation_"
for file in *.jl; do
  if [[ $file =~ Estimation_ ]]; then
    # Run the file using Julia interpreter
    julia "$file"
  fi
done

echo "Generating LaTeX Tables"

# Loop through all .jl files containing "new_layout"
for file in *.jl; do
  if [[ $file =~ new_layout ]]; then
    # Run the file using Julia interpreter
    julia "$file"
  fi
done

cd ../

#rm -rf Data/*

echo "All scripts finished running!"


