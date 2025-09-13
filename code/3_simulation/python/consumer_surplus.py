###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

# ##################################################################################################################################
# This file calculates consumer surpluses for amenities and airbnb rentals, as well as generating relevant plots and figures.
##################################################################################################################################

# Required library imports
import os
import json
import sys
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
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Defining runtime config (hyperparameters) and constants
if len(sys.argv) != 4:
    print("Error! No config file supplied.")
    print("Usage: python equilibrium_solver.py <config_file> <gamma_val> <use_B>")
    sys.exit(1)
    
print("Running with", sys.argv[1])

gamma_val = sys.argv[2]
value = str(gamma_val).zfill(3)

jax_config.update('jax_enable_x64', True)


# Loading config.json file to parse input
with open(str(sys.argv[1]), 'r') as f:
    config = json.load(f)

config['experiment_name'] = "final_model"
config['counterfactual'] = ""

EXPERIMENT_NAME_BASE = config['experiment_name'] + "_" + config["agg_type"]
if config['counterfactual'] != "":
    EXPERIMENT_NAME_BASE += "_" + config['counterfactual']


# Load binaries and start logger using files at path specified
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
P['no_airbnb'] = False
P['airbnb_tax_rate'] = 0
P['airbnb_extra_fee'] = 0
P['exo_amenities'] = False
P['airbnb_prices_exogenous'] = False
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

# Read price sensitivity
delta_r = P['delta_r']
exp_shares = 1 - P['Cobb_Douglas_housing']

# Determining which file path to use, and ensuring it is present in the file system.
if P['use_B']:
    filepath = '../../../data/simulation_results/gamma_B_' + value + '/counterfactuals/'
else:
    filepath = '../../../data/simulation_results/gamma_' + value + '/counterfactuals/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

runs_dir = filepath

# Creating inital surplus matrices
airbnb_consumer_surpluses = np.zeros((9, 4))
amenity_consumer_surpluses = np.zeros((9, 4))

################################################################################
# Run Consumer Surplus for Each Case
################################################################################
# Baseline
r_endo = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/equilibrium_objects" + "/r_endo.csv", header=None).to_numpy()
a_endo = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/equilibrium_objects" + "/a_endo.csv", header=None).to_numpy()

baseline_consumer_surplus = consumer_surplus(r_endo, a_endo, EVs_initial) / (P['w']) * 100
print("Baseline Consumer Surplus: ", baseline_consumer_surplus)


################################################################################
# AirBnB Tax Counterfactual
################################################################################

# For each tax rate, we read the results from our simulations to then calculate surplus values for each group
for tax in range(0, 9, 1):

    r = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/counterfactuals/airbnb_tax_proportional" + "/r_endo_airbnb_city_tax_proportional_" + str(tax/100) + ".csv", header=None).to_numpy()
    a = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/counterfactuals/airbnb_tax_proportional" + "/a_endo_airbnb_city_tax_proportional_" + str(tax/100) + ".csv", header=None).to_numpy()

    airbnb_tax_consumer_surplus = consumer_surplus(r, a, EVs_initial)/ (P['w']) * 100
    print("AirBnB Tax Rate " + str(tax/100) + " Consumer Surplus: " + str(airbnb_tax_consumer_surplus))
    airbnb_consumer_surpluses[tax] = np.hstack([tax/100, airbnb_tax_consumer_surplus.flatten()])


# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = airbnb_consumer_surpluses[:, 0] 
y = airbnb_consumer_surpluses[:, 1] 

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)  # Plot each column of y against x

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 1 Against AirBnB Tax")
plt.legend(["Group 1"]) 

# Save the plot as a PNG image
#fig.savefig(filepath + "airbnb_prop_tax_g1.png", dpi=300, bbox_inches="tight")
print("Plot saved as airbnb_prop_tax_g1.png")

# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = airbnb_consumer_surpluses[:, 0] 
y = airbnb_consumer_surpluses[:, 2]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 2 Against AirBnB Tax")
plt.legend(["Group 2"])

# Save the plot as a PNG image
#fig.savefig(filepath + "airbnb_prop_tax_g2.png", dpi=300, bbox_inches="tight")
print("Plot saved as airbnb_prop_tax_g2.png")


# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = airbnb_consumer_surpluses[:, 0]  
y = airbnb_consumer_surpluses[:, 3] 

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 3 Against AirBnB Tax")
plt.legend(["Group 3"]) 

# Save the plot as a PNG image
#fig.savefig(filepath + "airbnb_prop_tax_g3.png", dpi=300, bbox_inches="tight")
print("Plot saved as airbnb_prop_tax_g3.png")



# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = airbnb_consumer_surpluses[:, 0]
y = airbnb_consumer_surpluses[:, 1:4]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)  # Plot each column of y against x

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Against AirBnB Tax")
plt.legend(["Group 1", "Group 2", "Group 3"])

# Save the plot as a PNG image
#fig.savefig(filepath + "airbnb_prop_tax.png", dpi=300, bbox_inches="tight")
print("Plot saved as airbnb_prop_tax.png")




################################################################################
# Amenities Tax Counterfactual
################################################################################

# For each tax rate, we read the results from our simulations to then calculate surplus values for each group
for tax in range(0, 9, 1):
    try:
        r = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/counterfactuals/amenity_tax" + "/r_endo_amenity_tax_" + str(tax/100) + ".csv", header=None).to_numpy()
        a = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + value + "/counterfactuals/amenity_tax" + "/a_endo_amenity_tax_" + str(tax/100) + ".csv", header=None).to_numpy()

        print(a)

        amenity_tax_consumer_surplus = consumer_surplus(r, a, EVs_initial)/ (P['w']) * 100
        print(str(consumer_surplus(r, a, EVs_initial)))
        print("Amenity Tax Rate " + str(tax/100) + " Consumer Surplus: " + str(amenity_tax_consumer_surplus))
        amenity_consumer_surpluses[tax] = np.hstack([tax/100, amenity_tax_consumer_surplus.flatten()])
    except:
        print("Amenity Tax Rate " + str(tax/100) + " MISSING!")


# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = amenity_consumer_surpluses[:, 0]
y = amenity_consumer_surpluses[:, 1]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 1 Against Amenity Tax")
plt.legend(["Group 1"])  # Add legend for each y-axis data

# Save the plot as a PNG image
#fig.savefig(filepath + "amenity_tax_g1.png", dpi=300, bbox_inches="tight")
print("Plot saved as amenity_tax_g1.png")




# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = amenity_consumer_surpluses[:, 0]
y = amenity_consumer_surpluses[:, 2]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 2 Against Amenity Tax")
plt.legend(["Group 2"])

# Save the plot as a PNG image
#fig.savefig(filepath + "amenity_tax_g2.png", dpi=300, bbox_inches="tight")
print("Plot saved as amenity_tax_g2.png")



# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = amenity_consumer_surpluses[:, 0]
y = amenity_consumer_surpluses[:, 3]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Group 3 Against Amenity Tax")
plt.legend(["Group 3"])

# Save the plot as a PNG image
#fig.savefig(filepath + "amenity_tax_g3.png", dpi=300, bbox_inches="tight")
print("Plot saved as amenity_tax_g3.png")


# Extract x-axis and y-axis data
# Select column 0 for x-axis
# Select columns 1-3 for y-axis
x = amenity_consumer_surpluses[:, 0]
y = amenity_consumer_surpluses[:, 1:4]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Customizing plot labels and titles
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Plot of Consumer Surplus Against Amenity Tax")
plt.legend(["Group 1", "Group 2", "Group 3"])

# Save the plot as a PNG image
#fig.savefig(filepath + "amenity_tax.png", dpi=300, bbox_inches="tight")
print("Plot saved as amenity_tax.png")

plt.close()

# Saving surpluses as .csv files
np.savetxt(filepath + "CS_pp_airbnb_tax.csv", airbnb_consumer_surpluses, delimiter=",")
np.savetxt(filepath + "CS_pp_amenity_tax.csv", amenity_consumer_surpluses, delimiter=",")
