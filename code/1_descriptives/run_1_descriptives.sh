#!/bin/bash

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp

set -e

echo "Running 1_stylized_facts.R"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"
Rscript "1_stylized_facts.R"

echo "Running 2_Hazard_rate.do"
echo "$script_dir"
cd "$script_dir"
stata-mp -e do "2_Hazard_rate.do"

echo "Running 3_housing_market_outcomes.R"
echo "$script_dir"
cd "$script_dir"
Rscript "3_housing_market_outcomes.R"

echo "Running 4_graph_heuristics.do"
echo "$script_dir"
cd "$script_dir"
stata-mp -e do "4_graph_heuristics.do"

echo "Running 5_tenancy_status_analysis.do"
echo "$script_dir"
cd "$script_dir"
stata-mp -e do "5_tenancy_status_analysis.do"

echo "Running 6_clustering_table.py"
echo "$script_dir"
cd "$script_dir"
python3 "6_clustering_table.py"

echo "Running 7_rent_imputation_woz_tables.py"
echo "$script_dir"
cd "$script_dir"
python3 "7_rent_imputation_woz_tables.py"
