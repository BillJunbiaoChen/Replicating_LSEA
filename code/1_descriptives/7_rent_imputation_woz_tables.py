###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Generate TeX file for rent regressions in Appendix
## Author: Sriram Tolety
###############################################

import pandas as pd
import numpy as np
import os

# Read the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
xls = pd.ExcelFile('../../data/cbs/exports/230825_0500_Revision ECMA Housign Data - Summer 2023/housing_data.xlsx')


# Load relevant sheets
woz_transaction_df = pd.read_excel(xls, sheet_name='woz_value_transaction_value', header=None)
rent_predictions_df = pd.read_excel(xls, sheet_name='rent predictions', header=None)
rent_meter_predictions_df = pd.read_excel(xls, sheet_name='rent meter predictions', header=None)

def to_three_decimals(x):
    return "{:.3f}".format(float(x))

# Helper function to find R^2 and N dynamically
def find_r2_n_values(df, se_row_idx):
    r2_value = to_three_decimals(df.iloc[se_row_idx + 1, 2])  # R^2 value comes after the SE row
    n_value = to_three_decimals(df.iloc[se_row_idx + 2, 2]) # N value comes after the R^2 row
    return r2_value, n_value

# Extract values for WOZ transaction values (sheet 1)
woz_coeff = to_three_decimals(woz_transaction_df.iloc[3, 2])  # WOZ value coefficient
woz_se = to_three_decimals(woz_transaction_df.iloc[4, 2])     # Standard error of WOZ value
constant = to_three_decimals(woz_transaction_df.iloc[5, 2])   # Constant term
constant_se = to_three_decimals(woz_transaction_df.iloc[6, 2]) # Standard error of constant
# Find R^2 and N dynamically after the SE values
woz_r2, woz_n = find_r2_n_values(woz_transaction_df, 6)



rf_in_sample_coeff = to_three_decimals(rent_predictions_df.iloc[3, 1])
rf_out_sample_coeff = to_three_decimals(rent_predictions_df.iloc[3, 2])
lm_in_sample_coeff = to_three_decimals(rent_predictions_df.iloc[5, 3])
lm_out_sample_coeff = to_three_decimals(rent_predictions_df.iloc[5, 4])

# Standard Errors
rf_in_sample_coeff_se = to_three_decimals(rent_predictions_df.iloc[4, 1])
rf_out_sample_coeff_se = to_three_decimals(rent_predictions_df.iloc[4, 2])
lm_in_sample_coeff_se = to_three_decimals(rent_predictions_df.iloc[6, 3])
lm_out_sample_coeff_se = to_three_decimals(rent_predictions_df.iloc[6, 4])

# Constants
rf_in_sample_const = to_three_decimals(rent_predictions_df.iloc[7, 1])
rf_out_sample_const = to_three_decimals(rent_predictions_df.iloc[7, 2])
lm_in_sample_const = to_three_decimals(rent_predictions_df.iloc[7, 3])
lm_out_sample_const = to_three_decimals(rent_predictions_df.iloc[7, 4])

rf_in_sample_const_se = to_three_decimals(rent_predictions_df.iloc[8, 1])
rf_out_sample_const_se = to_three_decimals(rent_predictions_df.iloc[8, 2])
lm_in_sample_const_se = to_three_decimals(rent_predictions_df.iloc[8, 3])
lm_out_sample_const_se = to_three_decimals(rent_predictions_df.iloc[8, 4])

# R-squared
rf_in_sample_r2 = to_three_decimals(rent_predictions_df.iloc[9, 1])
rf_out_sample_r2 = to_three_decimals(rent_predictions_df.iloc[9, 2])
lm_in_sample_r2 = to_three_decimals(rent_predictions_df.iloc[9, 3])
lm_out_sample_r2 = to_three_decimals(rent_predictions_df.iloc[9, 4])

# Number of observations (N)
rf_in_sample_n = rent_predictions_df.iloc[10, 1]
rf_out_sample_n = rent_predictions_df.iloc[10, 2]
lm_in_sample_n = rent_predictions_df.iloc[10, 3]
lm_out_sample_n = rent_predictions_df.iloc[10, 4]




# Extract values for rent per square meter predictions (sheet 3)
rent_sqm_rf_in_sample_coeff = to_three_decimals(rent_meter_predictions_df.iloc[3, 1])
rent_sqm_rf_out_sample_coeff = to_three_decimals(rent_meter_predictions_df.iloc[3, 2])
rent_sqm_lm_in_sample_coeff = to_three_decimals(rent_meter_predictions_df.iloc[5, 3])
rent_sqm_lm_out_sample_coeff = to_three_decimals(rent_meter_predictions_df.iloc[5, 4])

# Standard Errors
rent_sqm_rf_in_sample_coeff_se = to_three_decimals(rent_meter_predictions_df.iloc[4, 1])
rent_sqm_rf_out_sample_coeff_se = to_three_decimals(rent_meter_predictions_df.iloc[4, 2])
rent_sqm_lm_in_sample_coeff_se = to_three_decimals(rent_meter_predictions_df.iloc[6, 3])
rent_sqm_lm_out_sample_coeff_se = to_three_decimals(rent_meter_predictions_df.iloc[6, 4])

# Constants
rent_sqm_rf_in_sample_const = to_three_decimals(rent_meter_predictions_df.iloc[7, 1])
rent_sqm_rf_out_sample_const = to_three_decimals(rent_meter_predictions_df.iloc[7, 2])
rent_sqm_lm_in_sample_const = to_three_decimals(rent_meter_predictions_df.iloc[7, 3])
rent_sqm_lm_out_sample_const = to_three_decimals(rent_meter_predictions_df.iloc[7, 4])

rent_sqm_rf_in_sample_const_se = to_three_decimals(rent_meter_predictions_df.iloc[8, 1])
rent_sqm_rf_out_sample_const_se = to_three_decimals(rent_meter_predictions_df.iloc[8, 2])
rent_sqm_lm_in_sample_const_se = to_three_decimals(rent_meter_predictions_df.iloc[8, 3])
rent_sqm_lm_out_sample_const_se = to_three_decimals(rent_meter_predictions_df.iloc[8, 4])

# R-squared
rent_sqm_rf_in_sample_r2 = to_three_decimals(rent_meter_predictions_df.iloc[9, 1])
rent_sqm_rf_out_sample_r2 = to_three_decimals(rent_meter_predictions_df.iloc[9, 2])
rent_sqm_lm_in_sample_r2 = to_three_decimals(rent_meter_predictions_df.iloc[9, 3])
rent_sqm_lm_out_sample_r2 = to_three_decimals(rent_meter_predictions_df.iloc[9, 4])

# Number of observations (N)
rent_sqm_rf_in_sample_n = rent_meter_predictions_df.iloc[10, 1]
rent_sqm_rf_out_sample_n = rent_meter_predictions_df.iloc[10, 2]
rent_sqm_lm_in_sample_n = rent_meter_predictions_df.iloc[10, 3]
rent_sqm_lm_out_sample_n = rent_meter_predictions_df.iloc[10, 4]


# WOZ Transaction LaTeX
woz_latex = f"""
\\begin{{table}}[H]
\\centering
\\footnotesize
\\caption{{Correlation between tax appraisal and transaction values.}}
\\label{{tab:woz_value_transaction_values}}
\\scalebox{{1}}{{
\\begin{{tabular}}{{@{{\\extracolsep{{0.5pt}}}}lcc}} 
\\toprule
 & \\multicolumn{{2}}{{c}}{{Transaction Value}} \\\\
\\cmidrule{{1-3}}
\\multicolumn{{1}}{{l}}{{WOZ Value}}     & \\multicolumn{{1}}{{c}}{{{woz_coeff}}} &\\multicolumn{{1}}{{c}}{{({woz_se})}}  \\\\
\\multicolumn{{1}}{{l}}{{Constant}}     & \\multicolumn{{1}}{{c}}{{{constant}}} &\\multicolumn{{1}}{{c}}{{ ({constant_se})}} \\\\
       \\cmidrule{{1-3}}
 $R^2$   & \\multicolumn{{2}}{{c}}{{{woz_r2}}}  \\\\
 $N$ & \\multicolumn{{2}}{{c}}{{{woz_n}}}  \\\\
\\bottomrule 
\\end{{tabular}} }}
\\begin{{minipage}}{{\\textwidth}}{{\\scriptsize Note: Table shows regression coefficients and fit of transaction values on tax appraisal (WOZ) values at the property level for Amsterdam 2005-2019. Standard errors in parenthesis.}}
\\end{{minipage}}
\\end{{table}} 
"""


# Imputation Results LaTeX
imputation_table = f"""
\\begin{{table}}[H]
\\centering
\\footnotesize
\\caption{{Imputation results.}}
\\label{{tab:imputation_results}}
  \\scalebox{{0.8}}{{

\\begin{{tabular}}{{@{{\\extracolsep{{-1.2pt}}}}lccccc}} 
\\toprule
&\\multicolumn{{5}}{{c}}{{In-sample fit}} \\\\
 & \\multicolumn{{2}}{{c}}{{Hedonic Model}} && \\multicolumn{{2}}{{c}}{{Random Forest}}\\\\ 
 \\cmidrule{{2-3}} \\cmidrule{{5-6}} \\\\
  & \\multicolumn{{1}}{{c}}{{Rental Prices}} & \\multicolumn{{1}}{{c}}{{ Price/m$^2$}} && \\multicolumn{{1}}{{c}}{{Rental Prices}} & \\multicolumn{{1}}{{c}}{{ Price/m$^2$}} \\\\ 
 \\\\[-1.8ex] 
  \\cmidrule{{1-6}}
$\\beta$     & {lm_in_sample_coeff} & {rent_sqm_lm_in_sample_coeff} && {rf_in_sample_coeff} & {rent_sqm_rf_in_sample_coeff} \\\\
     & ({lm_in_sample_coeff_se}) & ({rent_sqm_lm_in_sample_coeff_se}) && ({rf_in_sample_coeff_se}) & ({rent_sqm_rf_in_sample_coeff_se}) \\\\
 c    & {lm_in_sample_const} & {rent_sqm_lm_in_sample_const} && {rf_in_sample_const} & {rent_sqm_rf_in_sample_const} \\\\
     & ({lm_in_sample_const_se}) & ({rent_sqm_lm_in_sample_const_se}) && ({rf_in_sample_const_se}) & ({rent_sqm_rf_in_sample_const_se}) \\\\
       \\cmidrule{{1-6}}
 $R^2$   & {lm_in_sample_r2} & {rent_sqm_lm_in_sample_r2} && {rf_in_sample_r2} & {rent_sqm_rf_in_sample_r2} \\\\
 $N$ & {lm_in_sample_n} & {rent_sqm_lm_in_sample_n} && {rf_in_sample_n} & {rent_sqm_rf_in_sample_n} \\\\
     \\bottomrule
\\end{{tabular}}

\\begin{{tabular}}{{@{{\\extracolsep{{-1.2pt}}}}lccccc}} 
\\toprule
&\\multicolumn{{5}}{{c}}{{Out-of-sample fit}} \\\\
 & \\multicolumn{{2}}{{c}}{{Hedonic Model}} && \\multicolumn{{2}}{{c}}{{Random Forest}}\\\\ 
 \\cmidrule{{2-3}} \\cmidrule{{5-6}} \\\\
  & \\multicolumn{{1}}{{c}}{{Rental Prices}} & \\multicolumn{{1}}{{c}}{{ Price/m$^2$}} && \\multicolumn{{1}}{{c}}{{Rental Prices}} & \\multicolumn{{1}}{{c}}{{ Price/m$^2$}} \\\\ 
 \\\\[-1.8ex] 
  \\cmidrule{{1-6}}
$\\beta$     & {lm_out_sample_coeff} & {rent_sqm_lm_out_sample_coeff} && {rf_out_sample_coeff} & {rent_sqm_rf_out_sample_coeff} \\\\
     & ({lm_out_sample_coeff_se}) & ({rent_sqm_lm_out_sample_coeff_se}) && ({rf_out_sample_coeff_se}) & ({rent_sqm_rf_out_sample_coeff_se}) \\\\
 c    & {lm_out_sample_const} & {rent_sqm_lm_out_sample_const} && {rf_out_sample_const} & {rent_sqm_rf_out_sample_const} \\\\
     & ({lm_out_sample_const_se}) & ({rent_sqm_lm_out_sample_const_se}) && ({rf_out_sample_const_se}) & ({rent_sqm_rf_out_sample_const_se}) \\\\
       \\cmidrule{{1-6}}
 $R^2$   & {lm_out_sample_r2} & {rent_sqm_lm_out_sample_r2} && {rf_out_sample_r2} & {rent_sqm_rf_out_sample_r2} \\\\
 $N$ & {lm_out_sample_n} & {rent_sqm_lm_out_sample_n} && {rf_out_sample_n} & {rent_sqm_rf_out_sample_n} \\\\
\\bottomrule 
\\end{{tabular}} }}

\\begin{{minipage}}{{\\textwidth}}{{\\scriptsize Notes: Table shows regression coefficients and fit of imputed rental prices on observed rental prices at the property level. We do so for a linear hedonic regression and a random forest, and for two different data samples, the training sample (left panel) to assess in-sample fit, and the testing sample (right panel) to assess out-of-sample fit. Standard errors in parenthesis.}}
\\end{{minipage}}
\\end{{table}}  
"""

# Save the LaTeX content to respective files
with open("../../output/tables/woz_transaction_values.tex", "w") as f:
    f.write(woz_latex)

with open("../../output/tables/imputation_results.tex", "w") as f:
    f.write(imputation_table)

print("LaTeX files generated successfully!")
