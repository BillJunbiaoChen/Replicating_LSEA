###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

# ##################################################################################################################################
# This file sets up, configures, and executes the amenity tax counterfactual simulations for a range of counterfactual tax rates.
##################################################################################################################################

# Required library imports
import os
import sys
import json

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reading in the gamma elasticity value and parsing the parameter
print("Running with", sys.argv[1])
gamma_val = sys.argv[1]
value = str(gamma_val).zfill(3)

# Creating bootstrap results folder location if needed
filepath = '../../../data/simulation_results/gamma_B_' + value + '/counterfactuals/amenity_tax/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

print("Result CSVs will be stored at " + filepath)

# Iterations are in increments of 1% of the amenity tax rate, from 0 to 9%
for tax in range(0, 9, 1):

    #### B VERSION
    # ENDOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
    # Setting up config.json with proper flags and values
    with open('config.json', 'r+') as f:
        data = json.load(f)
        data['experiment_name'] = "final_model"
        data['no_airbnb'] = False
        data['airbnb_prices_exogenous'] = False
        data['homogenous_thetas'] = False
        data['exo_amenities'] = False
        data['sanity_checks'] = False
        data['old_version'] = False
        data['grouped_store'] = value
        data['grouped_run'] = True
        data['gamma'] = gamma_val
        data['use_B'] = True

        data['counterfactual'] = 'amenity_tax'
        data['airbnb_tax_rate'] = 0
        data["airbnb_extra_fee"] = 0
        data['amenity_tax_rate'] = tax/100
        data['radius'] = 0

        f.seek(0)
        f.truncate(0)
        json.dump(data, f, indent=4)

    os.system("python3 equilibrium_solver.py config.json")  


# Flushing and clearing the config.json to set up for future executions for the simulation / model. 
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data['no_airbnb'] = False
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = False
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-08
    data["lambda"] = 0.85

    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)
