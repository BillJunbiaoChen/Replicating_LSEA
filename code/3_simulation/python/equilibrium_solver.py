###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

# ##################################################################################################################################
# This file is the main simulation file. It drives the simulation, by first reading the configuration file (config.json)
# # Then after optionally executing sanity checks and perturbations, counterfactual generation, etc, this file calls \
# the main simulation functions as defined in main_functions.py and helper_functions.py
# Helper functions are called in order to save checkpoints and plots throughout the simulation.
# See the paper for more details.
##################################################################################################################################

# Required library imports
import sys
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
import os
from main_functions import outer_loop, tatonnement_prices, _set_constants, S_L, D_L, ED_L, utility, ED_S, Amenity_supply_exo, static_utility, calc_hotel_pop, D_S, tourist_guest_demand
from helper_functions import log_statistics, log_plots, load_checkpoint, write_checkpoint, load_binaries, write_config, write_eq_vals, _set_P_helper

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Defining runtime config (hyperparameters) and constants
if len(sys.argv) != 2:
    print("Error! No config file supplied.")
    print("Usage: python equilibrium_solver.py <config_file>")
    sys.exit(1)
    
print("Running with", sys.argv[1])
jax.config.update('jax_enable_x64', True)
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


# Loading config.json file to parse input
with open(str(sys.argv[1]), 'r') as f:
    config = json.load(f)

# Defining the experiment name based on configuration file flags and 
# user input for elasticity and boostrap vs non-bootsrap data
EXPERIMENT_NAME_BASE = config['experiment_name'] + "_" + config["agg_type"]
if config['counterfactual'] != "":
    EXPERIMENT_NAME_BASE += "_" + config['counterfactual']


EXPERIMENT_NAME = EXPERIMENT_NAME_BASE
if config["counterfactual"] == "airbnb_city_tax":
    EXPERIMENT_NAME += "_" + str(config["airbnb_extra_fee"])
if config["counterfactual"] == "airbnb_city_tax_proportional":
    EXPERIMENT_NAME += "_" + str(config["airbnb_tax_rate"])
if config["counterfactual"] == "amenity_tax":
    EXPERIMENT_NAME += "_" + str(config["amenity_tax_rate"])
if config['old_version']:
    EXPERIMENT_NAME += "_OLD_VERSION"
if config['new_solver']:
    EXPERIMENT_NAME += "_NEW_SOLVER"
if config['no_airbnb']:
    EXPERIMENT_NAME +=  "_no_airbnb"
if config['exo_amenities']:
    EXPERIMENT_NAME +=  "_exo_amenities"
if config['comment'] != "":
    EXPERIMENT_NAME +=  "_" + config['comment']


if config['endogenous_tourist_choices']:
    EXPERIMENT_NAME += "_endogenous_tourist_choices"
if config['airbnb_prices_exogenous']:
    EXPERIMENT_NAME += "_airbnb_prices_exogenous"
if config['homogenous_thetas']:
    EXPERIMENT_NAME += "_homogenous_thetas"
if config['constant_rent']:
    EXPERIMENT_NAME += "_constant_rent"
if config['static']:
    EXPERIMENT_NAME += "_static"
if config['myopic']:
    EXPERIMENT_NAME += "_myopic"
if not config['remove_students_from_supply']:
    EXPERIMENT_NAME += "_WITH_STUDENTS"
EXPERIMENT_NAME += "_gamma" + str(config["gamma"])

if config['use_B']:
    EXPERIMENT_NAME += "_B"

if config['intermediate_amenities'] != "":
    EXPERIMENT_NAME += "_" + "intermediateamenities_" + config['intermediate_amenities']
if config['radius']:
    EXPERIMENT_NAME += "_" + "radius_" + str(config['radius'])

print("EXPERIMENT_NAME_BASE")
print(EXPERIMENT_NAME_BASE)

# Defining value ranges for plots, and checkpoint folder to store checkpoints for simulations
AMENITY_NAMES = ["Tourism offices", "Restaurants", "Bars", "Food stores", "Non-food stores","Nurseries"]
AMENITY_RANGES = {
    "Tourism offices": {"x": (0, 1500, 4), "y": (0, 2000, 5)},
    "Restaurants": {"x": (0, 600, 4), "y": (0, 600, 4)},
    "Bars": {"x": (0, 300, 4), "y": (0, 300, 4)},
    "Food stores": {"x": (0, 300, 3), "y": (0, 400, 4)},
    "Non-food stores": {"x": (0, 1000, 6), "y": (0, 1500, 4)},
    "Nurseries": {"x": (250, 1750, 4), "y": (0, 2500, 6)}
}
CHECKPOINT_PREFIX = "Checkpoints/" + EXPERIMENT_NAME + "/"

# Load binaries and start logger using files at path specified.
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

P['radius'] = config['radius']
P['radius_iteration'] = config['radius_iteration']


P['counterfactual'] = config['counterfactual']

if config["counterfactual"] == "airbnb_city_tax":
    P['airbnb_extra_fee'] = config["airbnb_extra_fee"]
else:
    P['airbnb_extra_fee'] = 0

if config["counterfactual"] == "airbnb_city_tax_proportional":
    P['airbnb_tax_rate'] = config["airbnb_tax_rate"]
else:
    P['airbnb_tax_rate'] = 0

if config["counterfactual"] == "amenity_tax":
    P['amenity_tax_rate'] = config['amenity_tax_rate']
else:
    P['amenity_tax_rate'] = config['amenity_tax_rate']





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


# Loading the defined constant, flags, variables defined above into the simulation
EVs_initial = jnp.ones((int(P["tau_bar"])*(int(P["J"])+1),int(P["K"])))
Path(CHECKPOINT_PREFIX).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(logdir=f"runs/{EXPERIMENT_NAME}")
write_config(config, EXPERIMENT_NAME)
_set_constants((P, year, gb, combined_cluster, transition_prob))
random_key = jax.random.PRNGKey(round(config["rand_key"]))


def generate_concentric_spheres_point(random_key, initial_a, r1, r2):
    random_percentage = jax.random.uniform(random_key, minval=r1, maxval=r2) / 100

    # Generate a point p on the unit sphere
    random_key, subkey = jax.random.split(random_key)
    random_dirs = jax.random.ball(subkey, P['S'], p=jnp.inf, shape=(P['J'],))

    initial_a += initial_a * random_dirs * random_percentage
    return initial_a, random_key

if config['radius']:
    initial_a, random_key = generate_concentric_spheres_point(random_key, initial_a, config['radius']-1, config['radius'])
    print("perturbed amenities!")

if config['intermediate_amenities'] != "":
    intermediate_checkpoint = EXPERIMENT_NAME_BASE + "_" + config['intermediate_amenities'] + "_1e-13_gamma" + str(config["gamma"])
    print("./Checkpoints/" + intermediate_checkpoint + "/")
    _, _, intermediate_a_e, _, _, _ = load_checkpoint("final", "./Checkpoints/" + intermediate_checkpoint + "/")
    initial_a = intermediate_a_e


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




print("----------------------------------------")
print(f"RUNNING {config['counterfactual']} | {'EXO' if config['exo_amenities'] else 'ENDO'}")
print("----------------------------------------")

print(EXPERIMENT_NAME)

sub = EXPERIMENT_NAME.replace("_" + config['counterfactual'], '')
sub = sub.replace("_" + str(config['airbnb_tax_rate']), '')
print(sub)

if config['exo_amenities']:
    if config["counterfactual"] == "airbnb_entry":
        if config["no_airbnb"]:
            # set a_e, a_g = a_0^Endo and load EVs_0^Endo
            a_e = jnp.load(f"Checkpoints/{sub}/a_e_final.npy")
            a_g = a_e
        else:

            if config['new_solver']:
                match = "_" + str(config['agg_type']) + "_NEW_SOLVER_"
            else:
                match = "_" + str(config['agg_type']) + "_"

            sub = sub.replace(match, match + "no_airbnb_")
            print(sub)

            sub = sub.replace("_exo_amenities", "")
            print(sub)

            a_e = jnp.load(f"Checkpoints/{sub}/a_e_final.npy")
            a_g = a_e
else:
    if config["counterfactual"] == "airbnb_entry" and config["no_airbnb"] == False:

        if config['new_solver']:
            match = "_" + str(config['agg_type']) + "_NEW_SOLVER_"
        else:
            match = "_" + str(config['agg_type']) + "_"

        sub = sub.replace(match, match + "no_airbnb_")
        print(sub)

        a_e = jnp.load(f"Checkpoints/{sub}/a_e_final.npy")
        a_g = a_e
    if config["counterfactual"] == "amenity_city_tax":
        assert config["airbnb_tax_rate"] == 0, "airbnb tax should be 0 for amenity tax simulations"

        P["alpha_ks"] = jnp.concatenate([1/(1 + config['amenity_tax_rate']) * jnp.ones((P['K'] + 4, 1)), jnp.ones((7, 5))], axis=1) * P['alpha_ks']
        r_e, p_e, a_e, _, _, _ = load_checkpoint(
            "final",
            "Checkpoints/" + config['experiment_name'] + "_" + config["agg_type"] + "_baseline/"
        )
        a_g = a_e
        _set_constants((P, year, gb, combined_cluster, transition_prob))



if config["counterfactual"] == "airbnb_city_tax":
    assert config["airbnb_tax_rate"] == 0.0
    
    if config["airbnb_extra_fee"] == 10:
        r_e, p_e, a_e, _, _, _ = load_checkpoint(
            "final",
            "Checkpoints/" + "_".join(EXPERIMENT_NAME.split("_")[:-1]).replace("airbnb_city_tax", "baseline") + "/"
        )
        print("Loading from", "Checkpoints/" + "_".join(EXPERIMENT_NAME.split("_")[:-1]).replace("airbnb_city_tax", "baseline") + "/")
    else:
        r_e, p_e, a_e, _, _, _ = load_checkpoint(
            "final",
            "Checkpoints/" + "_".join(EXPERIMENT_NAME.split("_")[:-1] + [str(config["airbnb_extra_fee"] - 10)]) + "/"
        )
        print("Loading from", "Checkpoints/" + "_".join(EXPERIMENT_NAME.split("_")[:-1] + [str(config["airbnb_extra_fee"] - 10)]) + "/")
    a_g = a_e
elif config["counterfactual"] == "airbnb_city_tax_proportional":
    assert config["airbnb_extra_fee"] == 0.0

    r_e, p_e, a_e, _, _, _ = load_checkpoint(
            "final",
            "Checkpoints/" + sub + "/"
    )

    r_e = jnp.array(r_e)
    p_e = jnp.array(p_e)
    a_e = jnp.array(a_e)

    print("Loading from", "Checkpoints/" + sub + "/")
    a_g = a_e

# Optionally load checkpoint
checkpoint_to_load = config["checkpoint"] # Set this to appropriate number
if checkpoint_to_load != "":
    if config["counterfactual"] != "baseline":
        raise NotImplementedError("Checkpointing for non-baseline runs is not implemented yet.")
    print("Loading checkpoint...")
    r_e, p_e, a_e, EVs_g, dist_a, a_g = load_checkpoint(checkpoint_to_load, CHECKPOINT_PREFIX)
    checkpoint_to_load = int(checkpoint_to_load)
else:
    print("No checkpoint specified. Skipping...")
    checkpoint_to_load = 0

"""
Run experiment
"""

P["rent_meter"] = initial_r
prices_observed = p_e

_set_P_helper(P)


if P["sanity_checks"]:

    initial_p = p_e
    jax.debug.print("**** CONDUCTING SANITY CHECKS *****")

    jax.debug.print("airbnb: {}", not P['no_airbnb'])

    filepath = "../../../data/final/"
    filepath = filepath + 'gamma_033' + '/'

    panel_covariates_demand = pd.read_csv(filepath + "inputs/gebied_covariates_panel.csv")
    panel_covariates_demand = np.take(panel_covariates_demand, np.where(panel_covariates_demand['year'] == 2017)[0], axis=0)
    avg_squared_footage = np.exp(panel_covariates_demand['log_area_by_usage_1'])
    # prices
    true_rent = (panel_covariates_demand['unit_rent'] / avg_squared_footage).to_numpy().flatten()
    jax.debug.print("julia r: {}", true_rent)

    print("NEW Max difference in r " + str(np.linalg.norm(true_rent - initial_r, np.inf)))

    initial_r = true_rent

    jax.debug.print("Hotel Population: {}", calc_hotel_pop(initial_r, initial_p, initial_a, P))
    jax.debug.print("Observed Rental Prices: {}", initial_r)
    jax.debug.print("Observed Airbnb Prices: {}", initial_p)

    amenities = Amenity_supply_exo(initial_a, initial_p, true_rent, P['Pop_w_students'][:, 0:3], jnp.array(jnp.sum(initial_a, axis=1)))
    jax.debug.print("Amenities: {}", amenities)

    difference = np.mean(np.subtract(amenities, initial_a), axis=0)
    print(difference)
    for i in range(P['S']):
        print("Mean difference between the predicted amenity " + str(i) + " and the observed value:" + str(difference[i]))

    jax.debug.print("Long Term Housing Supply: {}", S_L(initial_r, initial_p))

    DL, EVs_g = D_L(initial_r, initial_a, EVs_initial)

    DL = DL[:P['J'],:]
    EDL = ED_L(initial_r, initial_p, DL)

    amenities = Amenity_supply_exo(initial_a, initial_p, initial_r, DL, jnp.array(jnp.sum(initial_a, axis=1)))
    jax.debug.print("Amenities: {}", amenities)

    jax.debug.print("Initial R: {}", initial_r)

    jax.debug.print("P['Pop']: {}", P['Pop'])
    jax.debug.print("Quantity Demanded @ log_r: {}", DL)
    jax.debug.print("Excess Demand @ log_r and log_p: {}", EDL)

    jax.debug.print("D_S: {}", D_S(initial_r, initial_p, initial_a))

    #jax.debug.print("ED_S: {}", ED_S(initial_r, initial_p, initial_a))
    jax.debug.print("Static Utility Group 0: {}", static_utility(initial_r, initial_a, 0)[0])
    jax.debug.print("Static Utility Group 0: {}", jnp.shape(static_utility(initial_r, initial_a, 0)[0]))
    
    tourist_demand_data_df = pd.read_csv(filepath + f"inputs/gebied_tourist_demand_covariates.csv")
    tourist_demand_data_df = tourist_demand_data_df[tourist_demand_data_df['gb_code'] != 0]
    tourist_demand_data_df = tourist_demand_data_df.sort_values(by=['gb_code'], ascending=True)
    total_population = tourist_demand_data_df['pop_tourist'].to_numpy().flatten()

    jax.debug.print("total_population: {}", total_population)
    jax.debug.print("Tourist Guest Demand: {}", tourist_guest_demand(initial_p, initial_a))

    print("NEW Max difference in tourist_giuest_demand " + str(np.linalg.norm(total_population - tourist_guest_demand(initial_p, initial_a), np.inf)))
    print("NEW Mean difference in tourist_giuest_demand " + str(np.mean(np.subtract(total_population, tourist_guest_demand(initial_p, initial_a)), axis=0)))
    

    panel_covariates_demand = pd.read_csv(filepath + "inputs/gebied_covariates_panel.csv")
    panel_covariates_demand = np.take(panel_covariates_demand, np.where(panel_covariates_demand['year'] == 2017)[0], axis=0)
    gebied_tourist_population = (panel_covariates_demand[['gb', 'pop_tourists_total']]
                             .drop_duplicates()
                             .sort_values(by=['gb'], ascending=True)
                             ['pop_tourists_total'].to_numpy())

    jax.debug.print("gebied_tourist_population: {}", gebied_tourist_population)

    jax.debug.print("Gamma 0: {}", P["gamma_0"])
    jax.debug.print("Gamma 1: {}", P["gamma_1"])
    jax.debug.print("Gamma 2: {}", P["gamma_2"])

    jax.debug.print("P.get('amenity_norm', 10**7): {}", P.get('amenity_norm', 10**7))

    jax.debug.print("**** PRESS ANY KEY TO PROCEED w/ SOLVING MODEL *****")
    sys.stdin.read(1)


if config['exo_amenities']:
    print("----------------------------------------")
    print("Running final market clearing...")
    print("----------------------------------------")
    r_e, p_e, delta_c, tol, i_tatonnement, EVs_g = tatonnement_prices(
        r_e,
        p_e,
        a_e,
        config['tatonnement_max_iter'],
        EVs_g,
        config['delta'],
        config['shrinkage'],
        config['delta_tol'],
        config['tatonnement_tol'],
        config['tatonnement_direct'],
    )

    print(f"Market cleared. max(ED_L, ED_S) == {tol}")
    write_checkpoint("final", CHECKPOINT_PREFIX, r_e, p_e, a_e, EVs_g, dist_a, a_g)
    if not config.get("no_log"):
        log_plots(writer, P, AMENITY_NAMES, AMENITY_RANGES, r_e, initial_r, p_e, initial_a, a_e, 0, False, EXPERIMENT_NAME)
else:
    if config['radius']:
        max_iters = 200
    else:
        max_iters = 700


    for s in tqdm(range(int(checkpoint_to_load/config['single_outer_loop_max_iter']), max_iters)):
        r_e, p_e, a_e, EVs_g, dist_a, a_g, i_outer, tatonnement_logs = outer_loop(
            a_e,
            r_e,
            p_e,
            config['single_outer_loop_max_iter'],
            config['lambda'],
            config['delta'],
            config['tatonnement_tol'],
            config['shrinkage'],
            config['delta_tol'],
            config['tatonnement_max_iter'],
            config['outer_loop_tol'],
            config['tatonnement_direct'],
            EVs_g,
            dist_a,
            a_g
        )

        jax.debug.print("R_E: {}", r_e)

        if s % config['single_outer_loop_max_iter'] == 0 and config.get('checkpointing', True):
            threading.Thread(target=write_checkpoint, args=((s+1)*config['single_outer_loop_max_iter'], CHECKPOINT_PREFIX, r_e, p_e, a_e, EVs_g, dist_a, a_g)).start()
            #if not config.get("no_log"):
            #    threading.Thread(target=log_plots, args=(writer, P, AMENITY_NAMES, AMENITY_RANGES, r_e, initial_r, p_e, initial_a, a_e, (s+1)*config['single_outer_loop_max_iter'], True)).start()

        #if not config.get("no_log"):
        #    threading.Thread(target=log_statistics, args=(writer, P, r_e, p_e, a_e, EVs_g, dist_a, a_g, tatonnement_logs, (s+1)*config['single_outer_loop_max_iter'])).start()

        if i_outer < config['single_outer_loop_max_iter']:
            print("Outer loop has converged. Terminating!")
            exit_i = (s+1)*config['single_outer_loop_max_iter'] + i_outer
            break
        exit_i = (s+1)*config['single_outer_loop_max_iter']

    print("----------------------------------------")
    print("Running final market clearing...")
    print("----------------------------------------")

    r_e, p_e, delta_c, tol, i_tatonnement, EVs_g = tatonnement_prices(
        r_e,
        p_e,
        a_e,
        config['tatonnement_max_iter'],
        EVs_g,
        config['delta'],
        config['shrinkage'],
        config['delta_tol'],
        config['tatonnement_tol'],
        config['tatonnement_direct'],
    )

    print(f"Market cleared. max(ED_L, ED_S) == {tol}")

    print("----------------------------------------")
    print(f"DONE! Creating final checkpoint. {EXPERIMENT_NAME} ({exit_i})")
    print("----------------------------------------")
    if not config.get("no_log"):
        log_plots(writer, P, AMENITY_NAMES, AMENITY_RANGES, r_e, initial_r, p_e, initial_a, a_e, exit_i, True, EXPERIMENT_NAME)
    write_checkpoint("final", CHECKPOINT_PREFIX, r_e, p_e, a_e, EVs_g, dist_a, a_g)

write_eq_vals("runs/" + EXPERIMENT_NAME + "/", r_e, p_e, a_e, EVs_g, not P['no_airbnb'], initial_a)



print(str(config['radius_iteration']))

if P["sanity_checks"]:
    print("************************************************************")
    print("R Equilibrium")
    print(r_e)
    print("************************************************************")

    print("P Equilibrium")
    print(p_e)
    print("************************************************************")

    print("A Equilibrium")
    print(a_e)
    print("************************************************************")

    print("************************************************************")
    print("Log R Equilibrium")
    print(np.log(r_e))
    print("************************************************************")

    print("Log P Equilibrium")
    print(np.log(p_e) + np.log(365))
    print("************************************************************")

    print("Mean Difference in Log P")
    print(np.mean(np.log(p_e) - np.log(prices_observed)))
    print("************************************************************")

