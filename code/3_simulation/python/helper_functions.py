###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file contains helper functions to log statistics, generate and save figures and plots, and save equilibria values.
##################################################################################################################################

# Required library imports
import json
import jax.numpy as jnp
import os
import csv
from scipy.stats import linregress
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from main_functions import D_L, D_S, S_L

global P

def _set_P_helper(tup):
    global P
    P = tup

"""
Logging functions for Tensorboard
"""
def log_statistics(writer, P, r_e, p_e, a_e, EVs_g, dist_a, a_g, tatonnement_logs, iteration):
    (delta_c, tol, i_tatonnement) = tatonnement_logs

    # Log convergence metrics
    writer.add_scalar('dist_a', dist_a, iteration)
    writer.add_scalar('tatonnment_tol', tol, iteration)
    writer.add_scalar('a_e-a_g', jnp.linalg.norm(a_e - a_g, ord=jnp.inf), iteration)

    # Log amenity prices
    writer.add_scalar('a_e[0][0]', a_e[0][0], iteration)
    writer.add_scalar('a_e[1][1]', a_e[1][1], iteration)
    writer.add_scalar('a_e[2][2]', a_e[2][2], iteration)
    writer.add_scalar('a_e[3][3]', a_e[3][3], iteration)
    writer.add_scalar('a_e[4][4]', a_e[4][4], iteration)
    writer.add_scalar('a_e[5][5]', a_e[5][5], iteration)
    # Log amenity prices
    writer.add_scalar('a_g[0][0]', a_g[0][0], iteration)
    writer.add_scalar('a_g[1][1]', a_g[1][1], iteration)
    writer.add_scalar('a_g[2][2]', a_g[2][2], iteration)
    writer.add_scalar('a_g[3][3]', a_g[3][3], iteration)
    writer.add_scalar('a_g[4][4]', a_g[4][4], iteration)
    writer.add_scalar('a_g[5][5]', a_g[5][5], iteration)

    # Log AirBnB prices
    writer.add_scalar('p_e[0]', p_e[0], iteration)

    # Log rent prices
    writer.add_scalar('r_e[0]', r_e[0], iteration)

    # Log tatonnement statistics
    writer.add_scalar('delta_c', delta_c, iteration)
    writer.add_scalar('tatonnement_i', i_tatonnement, iteration)

def log_plots(writer, P, amenities_names, amenities_ranges, r_eq_full, initial_r, p_eq_full, initial_a, a_eq_full, iteration, save_plot, filepath = ""):
    # Compare to observed rent
    model = linregress(initial_r, r_eq_full)
    
    r2m = model.rvalue**2
    coefs_m = model.slope, model.intercept

    rounded_r2 = round(r2m, 3)
    intercept_rounded = round(coefs_m[1], 3)
    slope_rounded = round(coefs_m[0], 3)
    fig, ax = plt.subplots()
    ax.scatter(initial_r, r_eq_full, label=False)
    ax.set_xlabel("Observed rent (per area unit & year)")
    ax.set_ylabel("Eq. (per area unit & year)")
    ax.set_title(f"Slope = {slope_rounded}, Intercept = {intercept_rounded}\n R2 = {rounded_r2}")
    ax.plot(initial_r, coefs_m[0] * initial_r + coefs_m[1])
    
    # Convert matplotlib figure to image
    if save_plot: 
        plt.savefig("runs/" + filepath + '/rent_plot.png')
    writer.add_figure("rent_plot", fig, iteration)
    plt.close(fig)

    # Compare to observed Airbnb prices
    model = linregress(P["p"], p_eq_full)
    r2m = model.rvalue**2
    coefs_m = model.slope, model.intercept
    rounded_r2 = round(r2m, 3)
    intercept_rounded = round(coefs_m[1], 3)
    slope_rounded = round(coefs_m[0], 3)
    fig, ax = plt.subplots()
    ax.scatter(P["p"], p_eq_full, label=False)
    ax.set_xlabel("Observed Airbnb prices (per night)")
    ax.set_ylabel("Eq. Airbnb prices (per night)")
    ax.set_title(f"Slope = {slope_rounded}, Intercept = {intercept_rounded}\n R2 = {rounded_r2}")
    ax.plot(P["p"], coefs_m[0] * P["p"] + coefs_m[1])

    # Convert matplotlib figure to image
    if save_plot: 
        plt.savefig("runs/" + filepath + '/airbnb_plot.png')
    writer.add_figure("airbnb_plot", fig, iteration)
    plt.close(fig)

    # Compare to observed amenities
    for i in range(P['S']):
        x = initial_a[:, i]
        y = a_eq_full[:, i]

        # Perform regression
        model = linregress(x, y)
        r2m = model.rvalue**2
        corr = model.rvalue
        coefs_m = model.slope, model.intercept
        std_err = model.stderr
        std_err_rounded = round(std_err, 3)
        rounded_r2 = round(r2m, 3)
        intercept_rounded = round(coefs_m[1], 3)
        slope_rounded = round(coefs_m[0], 3)
        corr_rounded = round(corr, 3)

        # Create data frame
        data = {'x': x, 'y': y}
        df = pd.DataFrame(data)

        # Set style
        sns.set_style("ticks", {"font.family": "Palatino", "text.color": "black"})
        fig, ax = plt.subplots()

        # Generate the plot with a confidence interval
        sns.regplot(x='x', y='y', data=df, ci=95, scatter_kws={'color': 'black', 's': 10}, line_kws={'color': 'black', 'linewidth': 1}, )
        plt.setp(ax.collections, alpha=0.05)

        # Spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)

        # Ticks
        ax.tick_params(width=1)

        # Tick ranges
        ax.set_xticks(np.linspace(*amenities_ranges[amenities_names[i]]['x']))
        ax.set_xlim([amenities_ranges[amenities_names[i]]['x'][0], amenities_ranges[amenities_names[i]]['x'][1]])
        ax.set_yticks(np.linspace(*amenities_ranges[amenities_names[i]]['y']))
        ax.set_ylim([amenities_ranges[amenities_names[i]]['y'][0], amenities_ranges[amenities_names[i]]['y'][1]])

        # Text
        ax.set_xlabel("Observed {}, 2017".format(amenities_names[i]), size='x-large')
        ax.set_ylabel("Simulated {}, 2017".format(amenities_names[i]), size='x-large')
        fig.suptitle(f"{amenities_names[i]}", size='xx-large', horizontalalignment='center', x=0.5)
        ax.set_title(f"{corr_rounded} correlation", horizontalalignment='center', x=0.5, fontsize='medium')
        ax.text(0.01, 0.95, f"Slope: {slope_rounded} ({std_err_rounded})\nR2: {rounded_r2}", verticalalignment='top', horizontalalignment='left', size='large', transform=ax.transAxes)
        
        if save_plot: 
            plt.savefig("runs/" + filepath + '/{}_plot.png'.format(amenities_names[i]))
        writer.add_figure("{} plot".format(amenities_names[i]), fig, iteration)
        plt.close(fig)


"""
Checkpointing functions
"""
def load_checkpoint(iteration, checkpoints_dir):
    r_e = jnp.load(checkpoints_dir + "r_e_{}.npy".format(iteration))
    p_e = jnp.load(checkpoints_dir + "p_e_{}.npy".format(iteration))
    a_e = jnp.load(checkpoints_dir + "a_e_{}.npy".format(iteration))
    EVs_g = jnp.load(checkpoints_dir + "EVs_g_{}.npy".format(iteration))
    dist_a = jnp.load(checkpoints_dir + "dist_a_{}.npy".format(iteration))
    a_g = jnp.load(checkpoints_dir + "a_g_{}.npy".format(iteration))
    return r_e, p_e, a_e, EVs_g, dist_a, a_g

def write_checkpoint(iteration, checkpoints_dir, r_e, p_e, a_e, EVs_g, dist_a, a_g):
    jnp.save(checkpoints_dir + "r_e_{}.npy".format(iteration), r_e)
    jnp.save(checkpoints_dir + "p_e_{}.npy".format(iteration), p_e)
    jnp.save(checkpoints_dir + "a_e_{}.npy".format(iteration), a_e)
    jnp.save(checkpoints_dir + "EVs_g_{}.npy".format(iteration), EVs_g)
    jnp.save(checkpoints_dir + "dist_a_{}.npy".format(iteration), dist_a)
    jnp.save(checkpoints_dir + "a_g_{}.npy".format(iteration), a_g)

def write_eq_vals(runs_dir, r_e, p_e, a_e, EVs_g, airbnb, initial_a):
    long_term_demand = D_L(r_e, a_e, EVs_g)
    population = long_term_demand[0]
    airbnb_demand = D_S(r_e, p_e, a_e)
    fraction_longterm = S_L(r_e, p_e)

    if airbnb:
        airbnb_demand = airbnb_demand[:-1]
    
    r_e = jnp.reshape(r_e, (22,1))
    p_e = jnp.reshape(p_e, (22,1))
    pop = jnp.reshape(population[:-1], (22,3))
    airbnb_demand = jnp.reshape(airbnb_demand, (22,1))
    fraction_longterm = jnp.reshape(fraction_longterm, (22,1))

    final_data = jnp.concatenate([r_e, p_e, a_e, pop, airbnb_demand, fraction_longterm], axis = 1)
    final_data = np.asarray(final_data)
    #print("Saving Final Equilibirum Values for R, P, A, D_L, airbnb_demand, fraction of long term housing in CSV at: " + runs_dir + "final_data.csv")
    #np.savetxt(runs_dir + "final_data.csv", final_data, delimiter=",")

    #np.savetxt(runs_dir + "r.csv", r_e, delimiter=",")
    #np.savetxt(runs_dir + "p.csv", p_e, delimiter=",")
    #np.savetxt(runs_dir + "DL.csv", pop, delimiter=",")
    #np.savetxt(runs_dir + "a.csv", a_e, delimiter=",")

    #np.savetxt(runs_dir + "init_a.csv", initial_a, delimiter=",")


    # The directory to save further files, as well as checking that python and julia 
    # simulation values converge, is dervied from our state space flags and variables.
    if P['grouped_run']:
        name = ""
        if P['exo_amenities']:
            name += "_exo"
        else:
            name += "_endo"

        if P['homogenous_thetas']:
            name += "_homogeneous"
        
        if P['no_airbnb']:
            name += "_no_airbnb"
        
        if P['counterfactual'] != "":
            name += "_" + P['counterfactual']

        if P["counterfactual"] == "airbnb_city_tax":
            name += "_" + str(P["airbnb_extra_fee"])
        if P["counterfactual"] == "airbnb_city_tax_proportional":
            name += "_" + str(P["airbnb_tax_rate"])
        if P["counterfactual"] == "amenity_tax":
            name += "_" + str(P["amenity_tax_rate"])
        
        if P['counterfactual'] == "airbnb_entry" and P['exo_amenities'] and P['no_airbnb'] == False:
            name = "_exo_a_endo_no_airbnb"
                
        name += ".csv"

        if P["counterfactual"] == "" or P["counterfactual"] == "airbnb_entry":
            if P['use_B']:
                filepath = '../../../data/simulation_results/gamma_B_' + P['grouped_store'] + '/equilibrium_objects/'
            else:
                filepath = '../../../data/simulation_results/gamma_' + P['grouped_store'] + '/equilibrium_objects/'
        elif P["counterfactual"] == "airbnb_city_tax_proportional":
            if P['use_B']:
                filepath = '../../../data/simulation_results/gamma_B_' + P['grouped_store'] + '/counterfactuals/airbnb_tax_proportional/'
            else:
                filepath = '../../../data/simulation_results/gamma_' + P['grouped_store'] + '/counterfactuals/airbnb_tax_proportional/'
        elif P["counterfactual"] == "amenity_tax":
            if P['use_B']:
                filepath = '../../../data/simulation_results/gamma_B_' + P['grouped_store'] + '/counterfactuals/amenity_tax/'
            else:
                filepath = '../../../data/simulation_results/gamma_' + P['grouped_store'] + '/counterfactuals/amenity_tax/'
        elif P['radius']:
            if P['use_B']:
                filepath = '../../../data/simulation_results/gamma_B_' + P['grouped_store'] + '/stability/amenities/'
            else:
                filepath = '../../../data/simulation_results/gamma_' + P['grouped_store'] + '/stability/amenities/'
        else:
            if P['use_B']:
                filepath = '../../../data/simulation_results/gamma_B_' + P['grouped_store'] + '/'
            else:
                filepath = '../../../data/simulation_results/gamma_' + P['grouped_store'] + '/'

        if P['radius']:
            baseline_values_r = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + P['grouped_store'] + "/equilibrium_objects" + "/r_endo.csv", header=None).to_numpy()
            baseline_values_p = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + P['grouped_store'] + "/equilibrium_objects" + "/p_endo.csv", header=None).to_numpy()
            baseline_values_a = pd.read_csv("../../../data/simulation_results/gamma_" + ("B_" if P['use_B'] else "") + P['grouped_store'] + "/equilibrium_objects" + "/a_endo.csv", header=None).to_numpy()

            r_diff = [np.abs(np.linalg.norm((baseline_values_r - r_e) / baseline_values_r * 100, np.inf).tolist())]
            p_diff = [np.abs(np.linalg.norm((baseline_values_p - p_e) / baseline_values_p * 100, np.inf).tolist())]
            a_diff = np.abs(np.linalg.norm((baseline_values_a - a_e) / baseline_values_a * 100, np.inf, axis=0).tolist())


            if os.path.exists(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '.csv') == False:
                open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', "w").close
            if os.path.exists(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '.csv') == False:
                open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', "w").close
            if os.path.exists(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '.csv') == False:
                open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', "w").close
            

            with open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', 'a', newline ='') as r_differences:
                write = csv.writer(r_differences)
                write.writerow(r_diff)
            with open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', 'a', newline ='') as p_differences:
                write = csv.writer(p_differences)
                write.writerow(p_diff)
            with open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '.csv', 'a', newline ='') as a_differences:
                write = csv.writer(a_differences)
                write.writerow(a_diff)
            

            r_diff = [np.abs(np.mean((baseline_values_r - r_e) / baseline_values_r * 100).tolist())]
            p_diff = [np.abs(np.mean((baseline_values_p - p_e) / baseline_values_p * 100).tolist())]
            a_diff = np.abs(np.mean((baseline_values_a - a_e) / baseline_values_a * 100, axis=0).tolist())

            if os.path.exists(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv') == False:
                open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', "w").close
            if os.path.exists(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv') == False:
                open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', "w").close
            if os.path.exists(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv') == False:
                open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', "w").close
            

            with open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', 'a', newline ='') as r_differences:
                write = csv.writer(r_differences)
                write.writerow(r_diff)
            with open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', 'a', newline ='') as p_differences:
                write = csv.writer(p_differences)
                write.writerow(p_diff)
            with open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_mean.csv', 'a', newline ='') as a_differences:
                write = csv.writer(a_differences)
                write.writerow(a_diff)


            r_diff = [np.abs(np.median((baseline_values_r - r_e) / baseline_values_r * 100)).tolist()]
            p_diff = [np.abs(np.median((baseline_values_p - p_e) / baseline_values_p * 100).tolist())]
            a_diff = np.abs(np.median((baseline_values_a - a_e) / baseline_values_a * 100, axis=0).tolist())

            if os.path.exists(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv') == False:
                open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', "w").close
            if os.path.exists(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv') == False:
                open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', "w").close
            if os.path.exists(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv') == False:
                open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', "w").close
            

            with open(filepath + 'r_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', 'a', newline ='') as r_differences:
                write = csv.writer(r_differences)
                write.writerow(r_diff)
            with open(filepath + 'p_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', 'a', newline ='') as p_differences:
                write = csv.writer(p_differences)
                write.writerow(p_diff)
            with open(filepath + 'a_differences_julia_' + 'radius_' + str(P['radius']) + '_median.csv', 'a', newline ='') as a_differences:
                write = csv.writer(a_differences)
                write.writerow(a_diff)


            return

        if os.path.exists(filepath + "r" + name) == False:
                open(filepath + "r" + name, "w+").close()
        if os.path.exists(filepath + "p" + name) == False:
            open(filepath + "p" + name, "w+").close()
        if os.path.exists(filepath + "DL" + name) == False:
            open(filepath + "DL" + name, "w+").close()
        if os.path.exists(filepath + "a" + name) == False:
            open(filepath + "a" + name, "w+").close()
        

        np.savetxt(filepath + "r" + name, r_e, delimiter=",")
        np.savetxt(filepath + "p" + name, p_e, delimiter=",")
        np.savetxt(filepath + "DL" + name, pop, delimiter=",")
        np.savetxt(filepath + "a" + name, a_e, delimiter=",")




"""
Loading estimations and parameter binaries
"""
def load_binaries(target_prefix):
    P = jnp.load("Binaries/" + target_prefix + "/" + "P.npy", allow_pickle = True)
    P = P.item()
    initial_r = jnp.load("Binaries/" + target_prefix + "/" + "initial_r.npy", allow_pickle = True)
    initial_a = jnp.load("Binaries/" + target_prefix + "/" + "initial_a.npy", allow_pickle = True)
    year = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.year.npy")
    gb = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.gb.npy")
    gb -= 1
    combined_cluster = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.combined_cluster.npy")
    transition_prob = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.transition_prob.npy")
    total_decision = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.total_decision.npy")
    total_by_tau = jnp.load("Binaries/" + target_prefix + "/" + "tau_trans_probs.total_by_tau.npy")
    return P, year, gb, combined_cluster, transition_prob, initial_a, initial_r

"""
Writes the config and binaries to the run folder
"""
def write_config(config, EXPERIMENT_NAME):
    with open(f"runs/{EXPERIMENT_NAME}/config.json", "w") as f:
        json.dump(config, f)