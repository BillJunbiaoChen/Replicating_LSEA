###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file sets up the stability csv from the stability runs for robustness analysis, and prepares the plotting data.
##################################################################################################################################


# %%
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

gamma_val = sys.argv[1]
value = str(gamma_val).zfill(3)

use_B_input = str(sys.argv[2])

if use_B_input == "True":
    gamma_string = "gamma_B_" + value
else:
    gamma_string = "gamma_" + value

# %%

amenities = pd.DataFrame()
airbnb = pd.DataFrame()
rent = pd.DataFrame()

import os
cwd = os.getcwd()
print(cwd)

# %%

for radius in range(2, 12, 2):
    amenities_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/a_differences_julia_radius_" + str(radius) + ".csv", header=None)
    airbnb_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/p_differences_julia_radius_" + str(radius) + ".csv", header = None)
    rent_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/r_differences_julia_radius_" + str(radius) + ".csv", header=None)

    amenities_difference.loc['mean'] = amenities_difference[-13:].mean()
    amenities_to_append = pd.Series(amenities_difference.loc['mean'])
    amenities = pd.concat([amenities, amenities_to_append.to_frame().T], ignore_index=True)


    airbnb_to_append = pd.Series(np.mean(airbnb_difference[-13:]))
    airbnb = pd.concat([airbnb, airbnb_to_append])

    rent = pd.concat([rent, pd.Series(np.mean(rent_difference[-13:]))])


# %%
print("Amenities Differences")
print(amenities)

# %%
print("Rent Differences")
print(rent)

# %%
print("Airbnb Differences")
print(airbnb)

# %%
target_prefix = "final_model_mean_" + gamma_string
P = jnp.load("python/Binaries/" + target_prefix + "/" + "P.npy", allow_pickle = True)
P = P.item()
initial_r = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_r.npy", allow_pickle = True)
initial_a = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_a.npy", allow_pickle = True)

# %%
np.max(initial_a)

# %%
np.divide(amenities, np.max(initial_a))

# %%
np.divide(rent, np.max(np.array(initial_r))) * 100

# %%
amenities = np.divide(amenities, np.max(np.array(initial_a))) * 100
airbnb = np.divide(airbnb, np.max(np.array(P['p']))) * 100
rent = np.divide(rent, np.max(np.array(initial_r))) * 100

# %%
average_values = np.zeros((6, 9))
average_values[0:, :] = np.zeros(9)
average_values[1:, 0] = list(range(2, 12, 2))
average_values[1:, 1:7] = amenities.to_numpy()
average_values[1:, 7] = airbnb.to_numpy().flatten()
average_values[1:, 8] = rent.to_numpy().flatten()


# %%
average_values

# %%
np.savetxt("../../data/simulation_results/"+gamma_string+"/stability/amenities/stability_averages.csv", average_values, delimiter=",")

# %%
observed_values = np.zeros((22, 8))
observed_values[:, 0:6] = initial_a
observed_values[:, 6] = np.array(P['p'])
observed_values[:, 7] = initial_r

# %%
np.savetxt("../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_a.csv", initial_a, delimiter=",")
np.savetxt("../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_r.csv", initial_r, delimiter=",")
np.savetxt("../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_p.csv", np.array(P['p']), delimiter=",")

# %%
observed_values




# %%

amenities = pd.DataFrame()
airbnb = pd.DataFrame()
rent = pd.DataFrame()

import os
cwd = os.getcwd()
print(cwd)

# %%

for radius in range(2, 12, 2):
    amenities_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/a_differences_julia_radius_" + str(radius) + "_mean.csv", header=None)
    airbnb_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/p_differences_julia_radius_" + str(radius) + "_mean.csv", header = None)
    rent_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/r_differences_julia_radius_" + str(radius) + "_mean.csv", header=None)

    amenities_difference.loc['mean'] = amenities_difference[-13:].mean()
    amenities_to_append = pd.Series(amenities_difference.loc['mean'])
    amenities = pd.concat([amenities, amenities_to_append.to_frame().T], ignore_index=True)


    airbnb_to_append = pd.Series(np.mean(airbnb_difference[-13:]))
    airbnb = pd.concat([airbnb, airbnb_to_append])

    rent = pd.concat([rent, pd.Series(np.mean(rent_difference[-13:]))])


# %%
print("Amenities Differences")
print(amenities)

# %%
print("Rent Differences")
print(rent)

# %%
print("Airbnb Differences")
print(airbnb)

# %%
target_prefix = "final_model_mean_" + gamma_string
P = jnp.load("python/Binaries/" + target_prefix + "/" + "P.npy", allow_pickle = True)
P = P.item()
initial_r = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_r.npy", allow_pickle = True)
initial_a = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_a.npy", allow_pickle = True)

# %%
np.max(initial_a)

# %%
np.divide(amenities, np.max(initial_a))

# %%
np.divide(rent, np.max(np.array(initial_r))) * 100

# %%
amenities = np.divide(amenities, np.max(np.array(initial_a))) * 100
airbnb = np.divide(airbnb, np.max(np.array(P['p']))) * 100
rent = np.divide(rent, np.max(np.array(initial_r))) * 100

# %%
average_values = np.zeros((6, 9))
average_values[0:, :] = np.zeros(9)
average_values[1:, 0] = list(range(2, 12, 2))
average_values[1:, 1:7] = amenities.to_numpy()
average_values[1:, 7] = airbnb.to_numpy().flatten()
average_values[1:, 8] = rent.to_numpy().flatten()


# %%
average_values

# %%
np.savetxt("../../data/simulation_results/"+gamma_string+"/stability/amenities/stability_averages_mean.csv", average_values, delimiter=",")









amenities = pd.DataFrame()
airbnb = pd.DataFrame()
rent = pd.DataFrame()

import os
cwd = os.getcwd()
print(cwd)

# %%

for radius in range(2, 12, 2):
    amenities_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/a_differences_julia_radius_" + str(radius) + "_median.csv", header=None)
    airbnb_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/p_differences_julia_radius_" + str(radius) + "_median.csv", header = None)
    rent_difference = pd.read_csv("../../data/simulation_results/"+gamma_string+"/stability/amenities/r_differences_julia_radius_" + str(radius) + "_median.csv", header=None)

    amenities_difference.loc['mean'] = amenities_difference[-13:].mean()
    amenities_to_append = pd.Series(amenities_difference.loc['mean'])
    amenities = pd.concat([amenities, amenities_to_append.to_frame().T], ignore_index=True)


    airbnb_to_append = pd.Series(np.mean(airbnb_difference[-13:]))
    airbnb = pd.concat([airbnb, airbnb_to_append])

    rent = pd.concat([rent, pd.Series(np.mean(rent_difference[-13:]))])

# %%
print("Amenities Differences")
print(amenities)

# %%
print("Rent Differences")
print(rent)

# %%
print("Airbnb Differences")
print(airbnb)

# %%
target_prefix = "final_model_mean_" + gamma_string
P = jnp.load("python/Binaries/" + target_prefix + "/" + "P.npy", allow_pickle = True)
P = P.item()
initial_r = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_r.npy", allow_pickle = True)
initial_a = jnp.load("python/Binaries/" + target_prefix + "/" + "initial_a.npy", allow_pickle = True)

# %%
amenities = np.divide(amenities, np.max(np.array(initial_a))) * 100
airbnb = np.divide(airbnb, np.max(np.array(P['p']))) * 100
rent = np.divide(rent, np.max(np.array(initial_r))) * 100

# %%
average_values = np.zeros((6, 9))
average_values[0:, :] = np.zeros(9)
average_values[1:, 0] = list(range(2, 12, 2))
average_values[1:, 1:7] = amenities.to_numpy()
average_values[1:, 7] = airbnb.to_numpy().flatten()
average_values[1:, 8] = rent.to_numpy().flatten()


# %%
average_values

# %%
np.savetxt("../../data/simulation_results/"+gamma_string+"/stability/amenities/stability_averages_median.csv", average_values, delimiter=",")