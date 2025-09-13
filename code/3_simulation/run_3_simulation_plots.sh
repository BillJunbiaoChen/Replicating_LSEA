#!/bin/bash

export PATH=$PATH:~/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:/Applications/Stata/StataSE.app/Contents/MacOS/
export PATH=$PATH:$HOME/Applications/Stata/StataMP.app/Contents/MacOS/
export PATH=$PATH:/usr/local/bin/stata-mp

set -e

echo "Generating Plots of Simulation"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
echo "Running plots/1a_counterfactual_heterogeneity.R"
Rscript "plots/1a_counterfactual_heterogeneity.R"

echo "$script_dir"
cd "$script_dir"
echo "Running plots/1b_counterfactual_airbnb_entry.R"
Rscript "plots/1b_counterfactual_airbnb_entry.R"


echo "$script_dir"
cd "$script_dir"

folder_path="../../output/figures/model_fit"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    # If it doesn't exist, create it
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi
chmod g+w $folder_path
chmod o+w $folder_path

echo "Running plots/2_model_fit.do"
stata-mp -e do "plots/2_model_fit.do" 152 True

echo "$script_dir"
cd "$script_dir"
echo "Running plots/3_CF_tax.do"
stata-mp -e do "plots/3_CF_tax.do" 152 True

echo "$script_dir"
cd "$script_dir"
echo "Running plots/stability_plots.do"
stata-mp -e do "plots/4_stability_plots.do" 152 True

echo "$script_dir"
cd "$script_dir"
echo "Running plots/4_generate_gini_table.py"
python3 "plots/5_generate_gini_table.py"

cd "$script_dir"
