###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

# ##################################################################################################################################
# This file estimates equilibrium staiblity by perturbing our equilibrium values at various radii as described in the paper. 
# It then calls the main equilibrium solver to then recalcualte the equilibria and estimate the norm difference between the two.
# See the paper for more details.
##################################################################################################################################

# Required library imports
import os
import sys
import json
import jax

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reading in the gamma elasticity value and parsing the parameter
print("Running with", sys.argv[1])
gamma_val = sys.argv[1]
value = str(gamma_val).zfill(3)

# Creating bootstrap results folder location if needed
filepath = '../../../data/simulation_results/gamma_B_' + value + '/stability/amenities/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

print("Result CSVs will be stored at " + filepath)

# Iterations are in increments of 1, from 1 - 10. This determines the radius of a norm ball used to perturb the equilibrium
for radius in range(1, 11, 1):

    key = jax.random.PRNGKey(round(radius))
    key, *subkeys = jax.random.split(key, 1000)
    subkeys = list(subkeys)

    for iteration in range(1, 11, 1):
        random_key = int(subkeys[iteration-1][0])
        
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
            data['airbnb_tax_rate'] = 0
            data["airbnb_extra_fee"] = 0
            data["amenity_tax_rate"] = 0
            data["outer_loop_tol"] = 1e-09
            data["lambda"] = 0.8

            data['counterfactual'] = 'stability'
            data['comment'] = 'julia_stability_run_' + str(iteration)
            data['radius'] = radius
            data['radius_iteration'] = iteration
            data['rand_key'] = random_key

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
    data["outer_loop_tol"] = 1e-08
    data["lambda"] = 0.85

    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0

    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)
