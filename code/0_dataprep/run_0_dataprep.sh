#!/bin/bash

# Run master file for cleaning export
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp

set -e
echo "$PATH"

echo "Running 0_master_dataprep_raw.do"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"
stata-se -e do "0_master_dataprep_raw.do" "0_master_dataprep_raw.log"

echo "Running 1_master_intermediate.R"
echo "$script_dir"
cd "$script_dir"
Rscript "1_master_intermediate.R"

echo "Running 2_master_dataprep_final.do"
echo "$script_dir"
cd "$script_dir"
stata-se -e do "2_master_dataprep_final.do"

echo "Running 3_gen_centroids_dist_mat.R"
echo "$script_dir"
cd "$script_dir"
Rscript "3_gen_centroids_dist_mat.R"

echo "Running 4_prep_eccp_paths.jl"
echo "$script_dir"
cd "$script_dir"
julia 4_prep_eccp_paths.jl