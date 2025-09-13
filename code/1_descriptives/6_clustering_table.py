###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Generate Tex file for demographic clusters
## Author: Sriram Tolety
###############################################

import pandas as pd
import numpy as np
import os

# Read the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
df = pd.read_excel('../../data/cbs/exports/Vrijgegeven220328_0500_Group Characteristics - April 2022/Group Characteristics - April 2022.xlsx')

# Function to format percentages
def format_percent(value):
    return f"{100*value:.2f}\\%"

# Function to format currency
def format_currency(value):
    return f"{value:,.2f}"

# Create the LaTeX table
latex_table = r"""
\begin{table}[!ht]
\centering
\footnotesize
\caption{Summary Statistics by Household Type}\label{fig: summary clusters}
\scalebox{0.825}{
\begin{tabular}{@{\extracolsep{10pt}}lcccccc} 
\toprule
& \multicolumn{2}{c}{Homeowners} & \multicolumn{2}{c}{Renters} & \multicolumn{2}{c}{Social Housing Tenants} \\
               \cmidrule{2-3} \cmidrule{4-5}  \cmidrule{6-7} \\[-1.5ex]
                 Group & \begin{tabular}[c]{@{}c@{}}Older\end{tabular}        & \begin{tabular}[c]{@{}c@{}}\end{tabular}          &  \begin{tabular}[c]{@{}c@{}}Younger\end{tabular}        & \begin{tabular}[c]{@{}c@{}}\end{tabular}     & \begin{tabular}[c]{@{}c@{}}Immigrant \end{tabular}     &
                 \begin{tabular}[c]{@{}c@{}}Dutch\end{tabular} \\
                 
                  & \begin{tabular}[c]{@{}c@{}}Families\end{tabular}        & \begin{tabular}[c]{@{}c@{}} Singles\end{tabular}           & \begin{tabular}[c]{@{}c@{}}Families\end{tabular}        & \begin{tabular}[c]{@{}c@{}} Students\end{tabular}     & \begin{tabular}[c]{@{}c@{}}Families \end{tabular}     &
                 \begin{tabular}[c]{@{}c@{}} Low Income\end{tabular} \\
               \cmidrule{2-3} \cmidrule{4-5}  \cmidrule{6-7} \\[-1.5ex]
"""

# Add rows to the table
latex_table += f"Age & {df['age'][0]:.2f} & {df['age'][1]:.2f} & {df['age'][2]:.2f} & {df['age'][3]:.2f} & {df['age'][4]:.2f} & {df['age'][5]:.2f}\\\\ [1ex]\n"
latex_table += f"Share with Children & {df['children'][0]:.2f} & {df['children'][1]:.2f} & {df['children'][2]:.2f} & {df['children'][3]:.2f} & {df['children'][4]:.2f}& {df['children'][5]:.2f}\\\\ [1ex]\n"
latex_table += r"   \midrule" + "\n"

latex_table += f"Share Low-Skilled & {format_percent(df['dskill1'][0])} & {format_percent(df['dskill1'][1])} & {format_percent(df['dskill1'][2])} & {format_percent(df['dskill1'][3])} & {format_percent(df['dskill1'][4])} & {format_percent(df['dskill1'][5])}\\\\ [1ex]\n"
latex_table += f"Share Medium-Skilled & {format_percent(df['dskill2'][0])} & {format_percent(df['dskill2'][1])} & {format_percent(df['dskill2'][2])} & {format_percent(df['dskill2'][3])} & {format_percent(df['dskill2'][4])} & {format_percent(df['dskill2'][5])}\\\\ [1ex]\n"
latex_table += f"Share High-Skilled & {format_percent(df['dskill3'][0])} & {format_percent(df['dskill3'][1])} & {format_percent(df['dskill3'][2])} & {format_percent(df['dskill3'][3])} & {format_percent(df['dskill3'][4])} & {format_percent(df['dskill3'][5])}\\\\ [1ex]\n"
latex_table += r"   \midrule" + "\n"

latex_table += f"Share Dutch Indies & {format_percent(df['dregion1'][0])} & {format_percent(df['dregion1'][1])} & {format_percent(df['dregion1'][2])} & {format_percent(df['dregion1'][3])} & {format_percent(df['dregion1'][4])} & {format_percent(df['dregion1'][5])}\\\\ [1ex]\n"
latex_table += f"Share Dutch & {format_percent(df['dregion2'][0])} & {format_percent(df['dregion2'][1])} & {format_percent(df['dregion2'][2])} & {format_percent(df['dregion2'][3])} & {format_percent(df['dregion2'][4])} & {format_percent(df['dregion2'][5])}\\\\ [1ex]\n"
latex_table += f"Share Non-Western & {format_percent(df['dregion3'][0])} & {format_percent(df['dregion3'][1])} & {format_percent(df['dregion3'][2])} & {format_percent(df['dregion3'][3])} & {format_percent(df['dregion3'][4])} & {format_percent(df['dregion3'][5])}\\\\ [1ex]\n"
latex_table += f"Share Western & {format_percent(df['dregion4'][0])} & {format_percent(df['dregion4'][1])} & {format_percent(df['dregion4'][2])} & {format_percent(df['dregion4'][3])} & {format_percent(df['dregion4'][4])} & {format_percent(df['dregion4'][5])}\\\\ [1ex]\n"
latex_table += r"   \midrule" + "\n"

latex_table += f"Household Income (\\euro) & {format_currency(df['imputed_disp_income'][0])} & {format_currency(df['imputed_disp_income'][1])} & {format_currency(df['imputed_disp_income'][2])} & {format_currency(df['imputed_disp_income'][3])} & {format_currency(df['imputed_disp_income'][4])} & {format_currency(df['imputed_disp_income'][5])}\\\\ [1ex]\n"
latex_table += f"Income Pctl. & {100*df['pctl_income'][0]:.2f} & {100*df['pctl_income'][1]:.2f} & {100*df['pctl_income'][2]:.2f} & {df['pctl_income'][3]:.2f} & {100*df['pctl_income'][4]:.2f} & {100*df['pctl_income'][5]:.2f}\\\\ [1ex]\n"
latex_table += f"Per Capita Income (\\euro) & {format_currency(df['imputed_income_per_person'][0])} & {format_currency(df['imputed_income_per_person'][1])} & {format_currency(df['imputed_income_per_person'][2])} & {format_currency(df['imputed_income_per_person'][3])} & {format_currency(df['imputed_income_per_person'][4])} & {format_currency(df['imputed_income_per_person'][5])}\\\\ [1ex]\n"
latex_table += f"Income Pctl. per Person & {100*df['pctl_income_per_person'][0]:.2f} & {100*df['pctl_income_per_person'][1]:.2f} & {100*df['pctl_income_per_person'][2]:.2f} & {100*df['pctl_income_per_person'][3]:.2f} & {100*df['pctl_income_per_person'][4]:.2f} & {100*df['pctl_income_per_person'][5]:.2f}\\\\ [1ex]\n"
latex_table += r"   \midrule" + "\n"

latex_table += f"Number of Households & {df['num_hh'][0]:,.0f} & {df['num_hh'][1]:,.0f} & {df['num_hh'][2]:,.0f} & {df['num_hh'][3]:,.0f} & {df['num_hh'][4]:,.0f} & {df['num_hh'][5]:,.0f}\\\\ [1ex]\n"

# Close the table
latex_table += r"""\bottomrule
\end{tabular}}
\legend{Table presents the groups resulting from k-means classification on mean demographic characteristics. We report average characteristics across households in each group. ``Low", ``medium", and ``high-skilled" correspond to high school or less, vocation/selective secondary education, and college and above, respectively. Group names are chosen to serve as an easy-to-remember label and are not an outcome of the data.}
\end{table}
"""

# Print the LaTeX table
print(latex_table)

with open('../../output/tables/clustered_demographics_group_data.tex', 'w') as f:
    f.write(latex_table)