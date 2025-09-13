###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Generate Tex file for Gini Differentiation
## Author: Sriram Tolety
###############################################
# 
# import re
import os
# File path
file_path = "../../output/tables/counterfactual_heterogeneity_differentiation_endo.tex"

# Read the file
with open(file_path, 'r') as file:
    content = file.read()

modified_content = content

modified_content = modified_content.replace(r"\begin{table}[!h]", r"\begin{table}[H]")

modified_content = modified_content.replace(r"\begin{tabular}", r"\scalebox{1}{\begin{tabular}")
modified_content = modified_content.replace(r"\end{tabular}", r"\end{tabular}}")

modified_content = modified_content.replace(r"\addlinespace", "")

footnote = r"""\legend*{\footnotesize Notes: Columns ``Homogeneous" and ``Heterogeneous" report the Gini index for each amenity sector: how concentrated the number of establishments in each sector is across locations. Higher values indicate most of the sector's establishments are clustered in a few locations. Column HE-HO reports the difference between the ``Heterogeneous" and ``Homogeneous" columns. Positive values in the HE-HO column indicate the spatial distribution of the amenity becomes more clustered across space when preferences are heterogeneous.}"""

modified_content = modified_content.replace(r"\end{table}", footnote + "\n" + r"\end{table}")

modified_content = modified_content.replace(r"\caption{\label{tab:heterogeneous vs homogenous - neighborhood differentiation}", 
                                            r"\caption{\label{tab:heterogeneous vs homogeneous - neighborhood differentiation}")

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(modified_content)

print(f"File has been modified and saved: {file_path}")