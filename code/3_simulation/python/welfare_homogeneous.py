###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file contains the welfare comparison for the heterogeneous-homogenous case
##################################################################################################################################

# Required library imports
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import threading
import jax
import jax.numpy as jnp
from jax.config import config as jax_config
import numpy as np
import pandas as pd
import jaxlib
from tensorboardX import SummaryWriter
from main_functions import *
from helper_functions import *
from main_functions import _set_constants
from sklearn import preprocessing

jax_config.update('jax_enable_x64', True)
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Define runtime config (hyperparameters) and constants
if len(sys.argv) != 4:
    print("Error! No config file supplied.")
    print("Usage: python equilibrium_solver.py <config_file> <gamma_val> <use_B>")
    sys.exit(1)
    

# Reading in the gamma elasticity value and parsing the parameter
print("Running with", sys.argv[1])
gamma_val = sys.argv[2]
value = str(gamma_val).zfill(3)


# Loading config.json file to parse input
with open(str(sys.argv[1]), 'r') as f:
    config = json.load(f)

# Defining runtime config (hyperparameters) and constants
config['counterfactual'] = ""

EXPERIMENT_NAME_BASE = config['experiment_name'] + "_" + config["agg_type"]
if config['counterfactual'] != "":
    EXPERIMENT_NAME_BASE += "_" + config['counterfactual']

#Load binaries and start logger using files at path specified.
use_B_input = str(sys.argv[3])

if use_B_input == "True":
    config['use_B'] = True
else:
    config['use_B'] = False

if config['counterfactual'] != "":
    if config['use_B']:
        P, year, gb, combined_cluster, transition_prob, initial_a, initial_r = load_binaries(EXPERIMENT_NAME_BASE.replace("_" + config['counterfactual'], '') + "_gamma_B_" + str(config["gamma"]))
    else:
        P, year, gb, combined_cluster, transition_prob, initial_a, initial_r = load_binaries(EXPERIMENT_NAME_BASE.replace("_" + config['counterfactual'], '') + "_gamma_" + str(config["gamma"]))
else:
    if config['use_B']:
        P, year, gb, combined_cluster, transition_prob, initial_a, initial_r = load_binaries(EXPERIMENT_NAME_BASE + "_gamma_B_" + str(config["gamma"]))
    else:
        P, year, gb, combined_cluster, transition_prob, initial_a, initial_r = load_binaries(EXPERIMENT_NAME_BASE + "_gamma_" + str(config["gamma"]))


# Calculating exact gamma value based on string input
P['gamma'] = float(config['gamma']) * -1 / 100
if config['gamma'] == "061":
    gamma = -1/1.65
    P['gamma'] = gamma
if config['gamma'] == "152":
    gamma = -1/.66
    P['gamma'] = gamma
if config['gamma'] == "333":
    gamma = -1/.3
    P['gamma'] = gamma
if config["gamma"] == "133":
    gamma = -1/.75
    P['gamma'] = gamma
if config["gamma"] == "116":
    gamma = -1/.86
    P['gamma'] = gamma
if config["gamma"] == "062":
    gamma = -1/1.61
    P['gamma'] = gamma
if config["gamma"] == "093":
    gamma = -1/1.07
    P['gamma'] = gamma

# Setting simulation parameters and flags as derived from config file or as needed
P['tol'] = config['EV_tol']
P['max_iter'] = 10 ** 6
P['no_airbnb'] = config['no_airbnb']
P['airbnb_tax_rate'] = config['airbnb_tax_rate']
P['airbnb_extra_fee'] = config['airbnb_extra_fee']
P['exo_amenities'] = config['exo_amenities']
P['airbnb_prices_exogenous'] = config['airbnb_prices_exogenous']
P['myopic'] = config['myopic']
P['static'] = config['static']
P['amenity_entry_costs_equilibrium_condition'] = config['amenity_entry_costs_equilibrium_condition']
P['remove_students_from_supply'] = config['remove_students_from_supply']
P['old_version'] = config['old_version']
P['new_solver'] = config['new_solver']
P['endogenous_tourist_choices'] = config['endogenous_tourist_choices']
P['grouped_run'] = config['grouped_run']
P['grouped_store'] = config['grouped_store']
P['homogenous_thetas'] = config['homogenous_thetas']
P['use_B'] = config['use_B']

# Variable setup
if config['myopic']:
    P['beta'] = 0
if config['static']:
    P['beta'] = 0
    P['delta_tau'] = jnp.zeros(3)
    P["gamma_0"] = jnp.zeros(3)
    P["gamma_1"] = jnp.zeros(3)
    P["gamma_2"] = jnp.zeros(3)
if P['old_version']:
    P['airbnb_prices_exogenous'] = True
    P['amenity_norm'] = 10**7
if config['homogenous_thetas']:
    total_sum = np.sum(P['Pop'])
    normalized_weights = P['Pop'] / total_sum
    print(normalized_weights)
    res = np.zeros((P['theta_resident'].shape[0],3))
    res = np.array(P['theta_resident'])
    for j in range(1, 7):  # Iterate through each row
       avg = np.dot(normalized_weights, P['theta_resident'][j, :])
       res[j, :] = [avg, avg, avg]
    P['theta_resident'] = jnp.array(res)
    print(P['theta_resident'])


P['constant_rent'] = config['constant_rent']
P['sanity_checks'] = config['sanity_checks']

EVs_initial = jnp.ones((int(P["tau_bar"])*(int(P["J"])+1),int(P["K"])))
random_key = jax.random.PRNGKey(round(config["radius"]))



a_e = initial_a
a_g = initial_a
p_e = P["p"]
r_e = initial_r
EVs_g = EVs_initial
dist_a = config['dist_a']



# Preparing the correct resident parameters based on the group (k) that we are calcualting utility for. 
X_resident_parameters = jnp.array(P['X_resident_parameters'], copy = True)[:, 1:]
X_resident_parameters = jnp.hstack((X_resident_parameters, jnp.full((22,1), 0)))
X_resident_parameters = jnp.hstack((X_resident_parameters, jnp.full((22,1), 0)))
X_resident_parameters = jnp.hstack((X_resident_parameters, jnp.full((22,1), 0)))
X_resident_parameters = jnp.hstack((X_resident_parameters, jnp.full((22,1), 0)))
columns_to_shift = X_resident_parameters[:, 39:41]
remaining_columns = X_resident_parameters[:, :39]
added_params = X_resident_parameters[:, 41:]
reordered_columns = jnp.concatenate([added_params, columns_to_shift], axis=1)
X_resident_parameters = jnp.concatenate([remaining_columns, reordered_columns], axis=1)

non_amentity_params = X_resident_parameters[:, 6:]
P['non_amentity_params'] = non_amentity_params


# Loading the defined constant, flags, variables defined above into the simulation
_set_constants((P, year, gb, combined_cluster, transition_prob))




################################################################################
# Read objects
################################################################################
# Homogeneous  
r_exo_homogeneous = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if config['use_B'] else "") + value + "/equilibrium_objects" + "/r_exo_homogeneous.csv", header=None).to_numpy()
p_exo_homogeneous = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if config['use_B'] else "") + value + "/equilibrium_objects" + "/p_exo_homogeneous.csv", header=None).to_numpy()
r_endo_homogeneous = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if config['use_B'] else "") + value + "/equilibrium_objects" + "/r_endo_homogeneous.csv", header=None).to_numpy()
a_endo_homogeneous = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if config['use_B'] else "") + value + "/equilibrium_objects" + "/a_endo_homogeneous.csv", header=None).to_numpy()
p_endo_homogeneous = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if config['use_B'] else "") + value + "/equilibrium_objects" + "/p_endo_homogeneous.csv", header=None).to_numpy()



# Read price sensitivity
delta_r = P['delta_r']
exp_shares = 1 - P['Cobb_Douglas_housing']

if P['use_B']:
    filepath = '../../../data/simulation_results/gamma_B_' + value + '/counterfactuals/'
else:
    filepath = '../../../data/simulation_results/gamma_' + value + '/counterfactuals/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

runs_dir = filepath

# Compte marginal utility of consumption
muc = -1 * np.multiply(1/(1 - P['beta']), (delta_r / (np.multiply(P['w'], exp_shares))))


print(muc)


total_sum = np.sum(P['Pop'])
normalized_weights = P['Pop'] / total_sum
print(normalized_weights)
res = np.zeros((P['theta_resident'].shape[0],3))
res = np.array(P['theta_resident'])
for j in range(1, 7):  # Iterate through each row
    avg = np.dot(normalized_weights, P['theta_resident'][j, :])
    res[j, :] = [avg, avg, avg]
P['theta_resident'] = jnp.array(res)
print(P['theta_resident'])
_set_constants((P, year, gb, combined_cluster, transition_prob))

EVs_initial = jnp.ones((int(P["tau_bar"])*(int(P["J"])+1),int(P["K"])))
## Exogenous amenities -- fixed at the observed level
CS_exo_homogeneous = np.zeros((P['K'],1))
CS_exo_homogeneous[:,0] = np.divide(welfare_households(r_exo_homogeneous, initial_a, EVs_initial), muc)
np.savetxt(runs_dir + "CS_exo_homogeneous.csv", CS_exo_homogeneous, delimiter=",")

print(CS_exo_homogeneous)

EVs_initial = jnp.ones((int(P["tau_bar"])*(int(P["J"])+1),int(P["K"])))
## Endogenous amenities
CS_endo_homogeneous = np.zeros((P['K'],1))
CS_endo_homogeneous[:,0] = np.divide(welfare_households(r_endo_homogeneous, a_endo_homogeneous, EVs_initial), muc)
np.savetxt(runs_dir + "CS_endo_homogeneous.csv", CS_endo_homogeneous, delimiter=",")

print(CS_endo_homogeneous)

print("***")
print(CS_exo_homogeneous - CS_endo_homogeneous)

print("########################### End of CF computation ##############################")
print("################################################################################")
