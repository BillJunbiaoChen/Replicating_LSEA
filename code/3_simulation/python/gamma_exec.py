###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file executes all variations of the model given a gamma elasticity parameter. 
# Calls equilibrium solver with different configurations in config.json. See paper for more details.
##################################################################################################################################

# Required library imports
import os
import sys
import json
import time
start = time.time()

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


print("Running with", sys.argv[1])
gamma_val = sys.argv[1]
value = str(gamma_val).zfill(3)

# Creating bootstrap results folder location if needed
filepath = '../../../data/simulation_results/gamma_B_' + value + '/equilibrium_objects/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

print("Result CSVs will be stored at " + filepath)


#### B VERSION
# ENDOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
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
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  


#### B VERSION
# ENDOGENEOUS AMENITIES, W/O AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
    data['no_airbnb'] = True
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = False
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  


#### B VERSION
# EXOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
    data['no_airbnb'] = False
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = True
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  
 

#### B VERSION
# EXOGENEOUS AMENITIES, W/O AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
    data['no_airbnb'] = True
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = True
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  



#### B VERSION
# ENDOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# HOMOGENEOUS THETAS - PREFERENCES HOMOGENEOUS ACROSS GROUPS
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
    data['no_airbnb'] = False
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = True
    data['exo_amenities'] = False
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.95
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  





#### B VERSION
# EXOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# HOMOGENEOUS THETAS - PREFERENCES HOMOGENEOUS ACROSS GROUPS
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data["counterfactual"] =  ""
    data['no_airbnb'] = False
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = True
    data['exo_amenities'] = True
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['counterfactual'] = ''
    data['radius'] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  


### AIRBNB_ENTRY CF
#### B VERSION
# EXOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data['no_airbnb'] = False
    data["counterfactual"] =  "airbnb_entry"
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = True
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.9
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)

os.system("python3 equilibrium_solver.py config.json")  


### AIRBNB_ENTRY CF
#### B VERSION
# ENDOGENEOUS AMENITIES, W/ AIRBNB IN MODEL
# Setting up config.json with proper flags and values
with open('config.json', 'r+') as f:
    data = json.load(f)
    data['experiment_name'] = "final_model"
    data['no_airbnb'] = False
    data["counterfactual"] =  ""
    data['airbnb_prices_exogenous'] = False
    data['homogenous_thetas'] = False
    data['exo_amenities'] = False
    data['sanity_checks'] = False
    data['old_version'] = False
    data['grouped_store'] = value
    data['grouped_run'] = True
    data['gamma'] = gamma_val
    data['use_B'] = True
    data['radius'] = 0
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.9
    data['airbnb_tax_rate'] = 0
    data["airbnb_extra_fee"] = 0
    data["amenity_tax_rate"] = 0
    data['radius_iteration'] = 0
    data['rand_key'] = 0
    data["checkpointing"] = True
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
    data["outer_loop_tol"] = 1e-10
    data["lambda"] = 0.85
    data["checkpointing"] = True
    f.seek(0)
    f.truncate(0)
    json.dump(data, f, indent=4)


#######AIRBNB PROP TAX CF
os.system("python3 airbnb_tax_cf_exec.py " + value)  


#######AMENITY TAX CF
os.system("python3 amenity_tax_cf_exec.py " + value)  


#######EQ STABILITY ANALYSIS
#os.system("python3 eq_stability_exec.py " + value)  

print('Full Run Took Approximately ', time.time()-start, ' Seconds!')
