###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Generate TeX file for WTP in Appendix
## Author: Sriram Tolety
###############################################

import pandas as pd
import numpy as np
import openpyxl
import os

# Read the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

N_main = 233772
N_myopic = 11132

total_table = np.zeros((18,7))


group_names = {1: "Older Families", 2: "Singles", 3: "Younger Families"}
amenity_names = ["Rent", "Touristic Amenities", "Restaurants", "Bars", 
                "Food Stores", "Non Food Stores", "Nurseries"]
rows = []


for group in range(1,4):

    ### GROUP 'group'
    with open('../../../output/estimates/var_cov_' + str(group) + '.csv', 'r') as file:
        lines = file.readlines()
    cleaned_lines = [line.replace('=', '').replace('"', '').strip() for line in lines]
    cleaned_data = [line.split(',') for line in cleaned_lines]
    df_cleaned = pd.DataFrame(cleaned_data)
    var_cov_matrix = df_cleaned.iloc[1:8, 1:8].astype(float)
    var_cov_beta = var_cov_matrix.values

    gmm_df = pd.read_csv('../../../output/estimates/gmm_demand_location_choice_estimates.csv', delimiter='\t')

    relevant_variables = [
        'log_rent_meter', 'log_amenity_1', 'log_amenity_2', 'log_amenity_3',
        'log_amenity_4', 'log_amenity_5', 'log_amenity_6', 'tau',
        'gamma_0_vec', 'gamma_1_vec', 'gamma_2_vec'
    ]

    filtered_df = gmm_df[gmm_df['Unnamed: 0'].isin(relevant_variables)]
    filtered_df = filtered_df[['Unnamed: 0', 'ivregress_g' + str(group)]]

    # Rename the columns for clarity
    filtered_df.columns = ['Variable', 'Value']

    filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')
    ordered_variables = [
        'log_rent_meter', 'log_amenity_1', 'log_amenity_2', 'log_amenity_3',
        'log_amenity_4', 'log_amenity_5', 'log_amenity_6', 'tau',
        'gamma_0_vec', 'gamma_1_vec', 'gamma_2_vec'
    ]
    ordered_df = filtered_df.set_index('Variable').reindex(ordered_variables).reset_index()

    beta = ordered_df['Value'].values

    K = len(beta)
    beta_price = beta[0]

    WTP = -beta[1:] / beta_price

    SE_WTP = np.zeros(6)

    for i in range(0, 6):
        part_1 = WTP[i]**2
        log_rent_meter_part = var_cov_beta[0, 0] / (beta[0]**2)
        own_var_part = var_cov_beta[i+1, i+1] / (beta[i+1]**2)
        covariance_part = 2 * (var_cov_beta[i+1, 0] / (beta[0] * beta[i+1]))

        SE_WTP[i] = np.sqrt(part_1 * (log_rent_meter_part + own_var_part - covariance_part))

    WTP = WTP[0:6]

    WTP_LB = WTP - 1.96 * SE_WTP
    WTP_UB = WTP + 1.96 * SE_WTP


    #Creating summary excel for demand
    coeffs = beta[0:7]
    group_name = group_names[group]
    for quality in range(7):
        if quality == 0:
            rows.append({
                "Household Type": group_name,
                "Characteristic": amenity_names[quality],
                "Coefficient": coeffs[quality],
                "WTP": -1.000
            })
        else:
            rows.append({
                "Household Type": group_name,
                "Characteristic": amenity_names[quality],
                "Coefficient": coeffs[quality],
                "WTP": WTP[quality-1]
            })


    ### GROUP 'group' - MYOPIC
    with open('../../../output/estimates/var_cov_static_' + str(group) + '.csv', 'r') as file:
        lines = file.readlines()
    cleaned_lines = [line.replace('=', '').replace('"', '').strip() for line in lines]
    cleaned_data = [line.split(',') for line in cleaned_lines]
    df_cleaned = pd.DataFrame(cleaned_data)
    var_cov_matrix = df_cleaned.iloc[1:8, 1:8].astype(float)
    var_cov_beta = var_cov_matrix.values

    gmm_df = pd.read_csv('../../../output/estimates/gmm_demand_location_choice_estimates_static.csv', delimiter='\t')

    relevant_variables = [
        'log_rent_meter', 'log_amenity_1', 'log_amenity_2', 'log_amenity_3',
        'log_amenity_4', 'log_amenity_5', 'log_amenity_6', 'tau',
        'gamma_0_vec', 'gamma_1_vec', 'gamma_2_vec'
    ]

    filtered_df = gmm_df[gmm_df['Unnamed: 0'].isin(relevant_variables)]
    filtered_df = filtered_df[['Unnamed: 0', 'ivregress_g'+str(group)]]

    # Rename the columns for clarity
    filtered_df.columns = ['Variable', 'Value']

    filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')

    ordered_variables = [
        'log_rent_meter', 'log_amenity_1', 'log_amenity_2', 'log_amenity_3',
        'log_amenity_4', 'log_amenity_5', 'log_amenity_6', 'tau',
        'gamma_0_vec', 'gamma_1_vec', 'gamma_2_vec'
    ]
    ordered_df = filtered_df.set_index('Variable').reindex(ordered_variables).reset_index()

    beta = ordered_df['Value'].values

    K = len(beta)
    beta_price = beta[0]

    WTP_myopic = -beta[1:] / beta_price

    SE_WTP_myopic = np.zeros(6)

    for i in range(0, 6):
        part_1 = WTP_myopic[i]**2
        log_rent_meter_part = var_cov_beta[0, 0] / (beta[0]**2)
        own_var_part = var_cov_beta[i+1, i+1] / (beta[i+1]**2)
        covariance_part = 2 * (var_cov_beta[i+1, 0] / (beta[0] * beta[i+1]))

        SE_WTP_myopic[i] = np.sqrt(part_1 * (log_rent_meter_part + own_var_part - covariance_part))

    WTP_myopic = WTP_myopic[0:6]

    WTP_LB_myopic = WTP_myopic - 1.96 * SE_WTP_myopic
    WTP_UB_myopic = WTP_myopic + 1.96 * SE_WTP_myopic

    Difference_Means = WTP - WTP_myopic
    Difference_SDs = np.sqrt((np.multiply(N_main, np.square(SE_WTP)) + np.multiply(N_myopic, np.square(SE_WTP_myopic))) / (N_main+N_myopic))
    t_test = np.divide(Difference_Means, Difference_SDs)

    total_table[(group-1)*6:(group)*6, 0] = WTP
    total_table[(group-1)*6:(group)*6, 1] = SE_WTP

    total_table[(group-1)*6:(group)*6, 2] = WTP_myopic
    total_table[(group-1)*6:(group)*6, 3] = SE_WTP_myopic

    total_table[(group-1)*6:(group)*6, 4] = Difference_Means
    total_table[(group-1)*6:(group)*6, 5] = Difference_SDs

    total_table[(group-1)*6:(group)*6, 6] = t_test



#### GENERATING DEMAND ESTIMATES SUMMARY EXCEL
sumary_demand_dynamic = pd.DataFrame(rows)
sumary_demand_dynamic['Coefficient'] = sumary_demand_dynamic['Coefficient'].apply(lambda x: '{:.3f}'.format(x))
sumary_demand_dynamic['WTP'] = sumary_demand_dynamic['WTP']



output_path = "../../../output/tables/demand_and_wtp_estimates.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    sumary_demand_dynamic.to_excel(writer, index=False, sheet_name='WTP Estimates')






###### GENERATING WTP TABLE TEX FILE
# Assume we have a numpy matrix 'total_table' with only float values
# Each row represents: WTP_Dynamic, sd_Dynamic, WTP_Static, sd_Static, Mean_Diff, sd_Diff, t-test

# List of amenities and groups (in order)
amenities = ["Touristic Amenities", "Restaurants", "Bars", "Food Stores", "Nonfood Stores", "Nurseries"]
groups = ["Older Families", "Singles", "Younger Families"]

def format_number(x):
    return f"{x:.4f}"

# Generate LaTeX table
latex_table = r"\begin{table}[H]" + "\n"
latex_table += r"    \centering" + "\n"
latex_table += r"    \caption{Comparison of dynamic and static estimates.}\label{tab: demand_estimation_locals-static v dynamic}" + "\n"
latex_table += r"    \scalebox{0.8}{" + "\n"
latex_table += r"    \begin{tabular}{llccccccc}" + "\n"
latex_table += r"        \toprule" + "\n"
latex_table += r"       \multicolumn{2}{c}{}  & \multicolumn{2}{c}{Dynamic}  & \multicolumn{2}{c}{Static}  & \multicolumn{3}{c}{Difference} \\" + "\n"
latex_table += r"       \cmidrule(l{3pt}r{3pt}){3-4} \cmidrule(l{3pt}r{3pt}){5-6} \cmidrule(l{3pt}r{3pt}){7-9}" + "\n"
latex_table += r"       Group & Amenity & WTP & sd & WTP & sd & Mean & sd  & t-test  \\" + "\n"
latex_table += r"       \midrule" + "\n"

for i, row in enumerate(total_table):
    group_index = i // len(amenities)
    amenity_index = i % len(amenities)
    
    if amenity_index == 0:
        group = groups[group_index]
        latex_table += r"       {0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} \\".format(
            group, amenities[amenity_index],
            format_number(row[0]), format_number(row[1]),
            format_number(row[2]), format_number(row[3]),
            format_number(row[4]), format_number(row[5]),
            format_number(row[6])
        ) + "\n"
    else:
        latex_table += r"        ~ & {0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} \\".format(
            amenities[amenity_index],
            format_number(row[0]), format_number(row[1]),
            format_number(row[2]), format_number(row[3]),
            format_number(row[4]), format_number(row[5]),
            format_number(row[6])
        ) + "\n"
    
    if amenity_index == len(amenities) - 1 and group_index < len(groups) - 1:
        latex_table += r"        \midrule" + "\n"

latex_table += r"        \bottomrule" + "\n"
latex_table += r"    \end{tabular}" + "\n"
latex_table += r"    }" + "\n"
latex_table += r"\end{table}"

print(latex_table)

with open("../../../output/tables/wtp_dynamic_vs_static.tex", "w") as f:
    f.write(latex_table)

print("LaTeX file generated successfully!")