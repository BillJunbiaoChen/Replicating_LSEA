###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file generates binaries of variables, constants, 
# and observed real-world values based on earlier estimation and data preperation files (See sections 0, 1, and 2)
##################################################################################################################################

# Required library imports
import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
import os
import json
import sys
from main_functions import static_utility, Amenity_supply_exo, _set_P

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Reading in the gamma elasticity value and parsing the parameter
filepath = "../../../data/final/"
gamma_val = str(sys.argv[1])
#filepath = filepath + 'gamma_' + gamma_val + '/'
use_B_input = str(sys.argv[2])

if use_B_input == "True":
    use_B = True
else:
    use_B = False

############################################################
# Utility Function DEFINITIONS
############################################################
def transform_reg_output(df_coefs: pd.DataFrame, reg_name: str, reg_name_after: str, K: int) -> pd.DataFrame:
    """Transforms the output of a regression model to a format that is compatible with the model.

    Args:
        df_coefs: A Pandas DataFrame containing the regression results.
        reg_name: The name of the regressor variable.
        reg_name_after: The name of the regressor variable after the transformation.
        K: The number of regressors.

    Returns:
        A Pandas DataFrame containing the transformed regression results.
    """

    # Rename the first column to `var`.
    df_coefs.rename(columns={"Unnamed: 0": "var"}, inplace=True)

    # Rename the regressor variables.
    for g in range(1, K + 1):
        df_coefs.rename(columns={f"{reg_name}{g}{reg_name_after}": f"g{g}"}, inplace=True)

    return df_coefs


def merge(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        merged_dict[key] = dict1[key]

    for key in dict2:
        if key in merged_dict:
            merged_dict[key] = dict2[key]
        else:
            merged_dict[key] = dict2[key]

    return merged_dict



############################################################
# GENERATING INTIAL_A, INITIAL_R BINARIES
############################################################

panel_covariates_demand = pd.read_csv(filepath + "inputs/gebied_covariates_panel.csv")
panel_covariates_demand = np.take(panel_covariates_demand, np.where(panel_covariates_demand['year'] == 2017)[0], axis=0)

needed_parameters = ["log_rent_meter", "log_amenity_1", "log_amenity_2", "log_amenity_3", "log_amenity_4", "log_amenity_5", "log_amenity_6", "dummy_gb1", "dummy_gb2", "dummy_gb3", "dummy_gb4", "dummy_gb5", "dummy_gb6", "dummy_gb7", "dummy_gb8", "dummy_gb9", "dummy_gb10", "dummy_gb11", "dummy_gb12", "dummy_gb13", "dummy_gb14", "dummy_gb15", "dummy_gb16", "dummy_gb17", "dummy_gb18", "dummy_gb19", "dummy_gb20", "dummy_gb21", "dummy_gb22", "dummy_year1", "dummy_year2", "dummy_year3", "dummy_year4", "dummy_year5", "dummy_year6", "dummy_year7", "dummy_year8", "dummy_year9", "dummy_year10", "dummy_year11", "log_social_housing", "log_area_by_usage_1"]
X_resident_parameters = panel_covariates_demand[needed_parameters]

true_rent = np.exp(panel_covariates_demand['log_rent_meter'])
initial_r = np.array(true_rent, dtype = float)

airbnb_population = panel_covariates_demand['pop_tourists_airbnb'].to_numpy()

amenities_data_mat = np.take(panel_covariates_demand, np.where(panel_covariates_demand['year'] == 2017)[0], axis= 0)[["amenity_1","amenity_2","amenity_3","amenity_4","amenity_5","amenity_6"]]
initial_a = np.array(amenities_data_mat, dtype = float)

print("************************************************************")
print("************************************************************")
print("************************************************************")
print("Successfuly Generated Initial A, Initial R Binaries!")



############################################################
# GENERATING TAU TRANSITION BINARIES
############################################################
J = 22
tau_trans_probs = pd.read_csv(filepath + "inputs/tau_transition_probs_all.csv")
tau_trans_probs = np.take(tau_trans_probs, np.where(tau_trans_probs['year'] == 2017)[0], axis=0)
tau_trans_probs['gb'] = tau_trans_probs['gb'].replace(0, J+1)
tau_trans_probs = tau_trans_probs.sort_values(by=['combined_cluster', 'gb'])

print("Successfuly Generated Tau Transition Binaries!")


############################################################
# GENERATING P BINARIES
############################################################

config = {
    'experiment_name' : "final_model",
    "counterfactual": "",
    "comment": "1e-13",
    "radius": 0,
    "seed": 0,
    "airbnb_tax_rate": 0.0,
    "airbnb_extra_fee": 0,
    "no_airbnb": True,
    "airbnb_outside_option": False,
    "exo_amenities": True,
    "agg_type": "mean",
    "checkpoint": "",
    "dist_a": 100,
    "gamma": -1 * int(gamma_val),
    "single_outer_loop_max_iter": 5,
    "lambda": 0.95,
    "delta": 0.1,
    "tatonnement_tol": 1e-4,
    "shrinkage": 0.98,
    "delta_tol": 1e-20,
    "tatonnement_max_iter": 1e5,
    "outer_loop_tol": 5e-5,
    "tatonnement_direct": False,
    "EV_tol": 1e-13,
    "no_log": False,
    "checkpointing": True
}


P = {
    'K' : 4,
    'S' : 6,
    'J' : 22,
    'T' : 11,
    'tau_bar' : 2,
    "radius": 0,
    "seed": 0,
    "use_B": use_B
}
    
# Calculating exact gamma value based on string input
if gamma_val == "061":
    gamma = -1/1.65
    P['gamma'] = gamma
elif gamma_val == "152":
    gamma = -1/.66
    P['gamma'] = gamma
elif gamma_val == "333":
    gamma = -1/.3
    P['gamma'] = gamma
elif gamma_val == "133":
    gamma = -1/.75
    P['gamma'] = gamma
elif gamma_val == "116":
    gamma = -1/.86
    P['gamma'] = gamma
elif gamma_val == "062":
    gamma = -1/1.61
    P['gamma'] = gamma
elif gamma_val == "093":
    gamma = -1/1.07
    P['gamma'] = gamma
else:
    config['gamma'] = float(gamma_val) * -1 / 100
    gamma = config['gamma']

experiment_name_base = config["experiment_name"]
experiment_name_counterfactual = experiment_name_base + "_" + config["agg_type"]
if config["counterfactual"] != "":
    experiment_name_counterfactual = experiment_name_counterfactual + "_" + config["counterfactual"]

experiment_name_counterfactual = experiment_name_counterfactual + '_gamma_' 

if P['use_B']:
    experiment_name_counterfactual = experiment_name_counterfactual + "B_" + gamma_val
else:
    experiment_name_counterfactual = experiment_name_counterfactual + gamma_val

if P['use_B']:
    gamma_string = "gamma_B_" + gamma_val
else:
    gamma_string = "gamma_" + gamma_val

directory_path = "../../../data/simulation_results/"+gamma_string+"/equilibrium_objects/"

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
else:
    print(f"Directory already exists: {directory_path}")

################################################################################
# Read data in
################################################################################

#### DEMAND ESTIMATES
# Define names of relevant variables
amenity_vars = ["log_tourism_offices", "log_restaurants", "log_bars", "log_food_stores", "log_nonfood_stores","log_nurseries"]
amenity_names = ["Tourism offices", "Restaurants", "Bars", "Food stores", "Non-food stores","Nurseries"]
amenity_vars_numeric = ["log_amenity_" + str(s) for s in range(1, P['S'] + 1)]
controls_vars = ["log_social_housing", "log_area_by_usage_1"]



# Read the file with estimates and clean data
demand_estimates = pd.read_csv(filepath + f"estimates/ivreg_demand_location_choice_estimates.csv", delimiter="\t")
demand_estimates = transform_reg_output(demand_estimates, "ivregress_g", "", 3)
demand_estimates = demand_estimates.loc[1:]
demand_estimates["var"] = demand_estimates[demand_estimates.columns[0]].astype(str)
# Parse the `g$g` columns as floats.
for g in range(1, 4):
    demand_estimates[f"g{g}"] = pd.to_numeric(demand_estimates[f"g{g}"], errors="coerce")

list_order_params = [
    "log_rent_meter",
    "log_amenity_1",
    "log_amenity_2",
    "log_amenity_3",
    "log_amenity_4",
    "log_amenity_5",
    "log_amenity_6",
    "dummy_gb1",
    "dummy_gb2",
    "dummy_gb3",
    "dummy_gb4",
    "dummy_gb5",
    "dummy_gb6",
    "dummy_gb7",
    "dummy_gb8",
    "dummy_gb9",
    "dummy_gb10",
    "dummy_gb11",
    "dummy_gb12",
    "dummy_gb13",
    "dummy_gb14",
    "dummy_gb15",
    "dummy_gb16",
    "dummy_gb17",
    "dummy_gb18",
    "dummy_gb19",
    "dummy_gb20",
    "dummy_gb21",
    "dummy_gb22",
    "dummy_year1",
    "dummy_year2",
    "dummy_year3",
    "dummy_year4",
    "dummy_year5",
    "dummy_year6",
    "dummy_year7",
    "dummy_year8",
    "dummy_year9",
    "dummy_year10",
    "dummy_year11",
    "tau",
    "gamma_0_vec",
    "gamma_1_vec",
    "gamma_2_vec",
    "log_social_housing",
    "log_area_by_usage_1"
]


# Get coefficients
rent_coefs = demand_estimates[demand_estimates['var'] == 'log_rent_meter'][['g1', 'g2', 'g3']].to_numpy()[0]

amenity_coefs = demand_estimates[demand_estimates['var'].isin(amenity_vars_numeric)][['g1', 'g2', 'g3']].to_numpy()
loc_FEs = demand_estimates[demand_estimates['var'].isin([f'dummy_gb{j}' for j in range(1, P['J'] + 1)])][['g1', 'g2', 'g3']].to_numpy()
time_FEs = demand_estimates[demand_estimates['var'].isin([f'dummy_year{t}' for t in range(1, P['T'] + 1)])][['g1', 'g2', 'g3']].to_numpy()
controls_coefs = demand_estimates[demand_estimates['var'].isin(controls_vars)][['g1', 'g2', 'g3']].to_numpy()
gamma_1_coef = np.array(demand_estimates[demand_estimates['var'] == 'gamma_1_vec'][['g1', 'g2', 'g3']].values)
gamma_0_coef = np.array(demand_estimates[demand_estimates['var'] == 'gamma_0_vec'][['g1', 'g2', 'g3']].values)
gamma_2_coef = np.array(demand_estimates[demand_estimates['var'] == 'gamma_2_vec'][['g1', 'g2', 'g3']].values)
tau_coef = np.array(demand_estimates[demand_estimates['var'] == 'tau'][['g1', 'g2', 'g3']].values)
controls_coefs_w_const = np.append(controls_coefs, np.ones(shape=(len(controls_coefs), 1)), axis=1)


theta_resident = np.array(demand_estimates)[:]
order_dict = {string: i for i, string in enumerate(list_order_params)}
indices = [order_dict.get(val, None) for val in theta_resident[:, 0]]
theta_resident = theta_resident[np.argsort(indices)]
theta_resident = theta_resident[:, :4]


#### HOUSING SUPPLY ESTIMATES
# Read in a file with estimated coefficients
housing_supply_parameters = pd.read_csv(filepath + f"estimates/housing_supply_estimates.csv")

alpha_housing_supply = housing_supply_parameters["alpha"]


#### AMENITY ESTIMATES
# Amenity supply estimates
gamma_amenity = -1 * gamma
gamma_amenity_val = f"{gamma_amenity:.2f}"
if P['use_B']:
    amenity_estimates_df = pd.read_csv(filepath + f"estimates/B_amenity_supply_estimation" + '_gamma_' + gamma_amenity_val + ".csv")
else:
    amenity_estimates_df = pd.read_csv(filepath + f"estimates/amenity_supply_estimation" + '_gamma_' + gamma_amenity_val + ".csv")

# Retrieve the coefficients
amenity_estimates = np.array(amenity_estimates_df['x1'])

alpha_est_range = range((P['K']+3)*P['S'])
alpha_estimates = amenity_estimates.take(alpha_est_range)
alpha_estimates = alpha_estimates.reshape(P['K']+3, P['S'], order='F')

location_FE_amenity_supply_range = range((P['K']+3)*P['S'], (P['K']+3)*P['S']+P['J']-1)
location_FE_amenity_supply = amenity_estimates.take(location_FE_amenity_supply_range)
location_FE_amenity_supply = np.append(0, location_FE_amenity_supply)

time_FE_amenity_supply = amenity_estimates[(P['K']+3)*P['S']+P['J'] - 1:]
time_FE_amenity_supply = np.append(time_FE_amenity_supply, 0)
time_FE_y_amenity_supply = time_FE_amenity_supply[-2]

# Prepare residuals
if P['use_B']:
    df_resid = pd.read_csv(filepath + f"estimates/B_amenity_supply_residuals" + '_gamma_' + gamma_amenity_val + ".csv")
else:
    df_resid = pd.read_csv(filepath + f"estimates/amenity_supply_residuals" + '_gamma_' + gamma_amenity_val + ".csv")
amenity_norm = df_resid['x5'][0]
df_resid.rename(columns={'x1': 's', 'x2': 'gb', 'x3': 'year', 'x4': 'resid'}, inplace=True)
df_resid = df_resid[df_resid['year'] == 2017]
df_resid.sort_values(by=['s', 'gb', 'year'], inplace=True)
amenity_supply_residuals = np.array(df_resid['resid']).reshape(P['J'], P['S'], order='F').T



#### Tourist demand
# Read in estimates
tourist_demand_estimates_df = pd.read_csv(filepath + f"estimates/tourist_demand_estimates.csv")
tourist_demand_price_coef = np.array(tourist_demand_estimates_df['b'][0])
tourist_demand_amenity_coef = np.array(tourist_demand_estimates_df['b'][1:-1])
tourist_demand_controls = np.array(tourist_demand_estimates_df['b'])
tourist_demand_controls = tourist_demand_controls[-1]


# Get expenditure shares
exp_shares = np.array(pd.read_csv(filepath + f"inputs/expenditure_shares.csv")['exp_sh_c'])
cobb_douglas_housing = 1 - exp_shares[0:6]
all_cons_share_tourists = exp_shares[-1]


# Annual income
income = pd.read_csv(filepath + f"inputs/annual_income.csv")
yearly_disp_income = (income[income['year'] == 2017]
                      .drop_duplicates()
                      .sort_values(by=['combined_cluster'], ascending=True)
                      ['disposable_income'].to_numpy())

# Income data
income_endo_types = yearly_disp_income[:3]
income_exo_types = yearly_disp_income[3:]


# Type characteristics
pop_w_tourists = pd.read_csv(filepath + f"inputs/gebied_population_counts_panel.csv")
pop_w_tourists = pop_w_tourists[pop_w_tourists['year'] == 2017]
type_gb_counts = (pop_w_tourists[['combined_cluster', 'gb', 'num_hh']].drop_duplicates()
                  .sort_values(by=['combined_cluster', 'gb'], ascending=True))

type_gb_counts = type_gb_counts[type_gb_counts.combined_cluster != 7]
type_gb_counts = type_gb_counts['num_hh'].to_numpy().reshape(P['J'] + 1, P['K'] + 2, order = 'F')
Pop = type_gb_counts.sum(axis=0)[0:3]
type_gb_counts = type_gb_counts[0:-1]
type_counts = type_gb_counts.sum(axis=0)
type_counts_inner = type_gb_counts[:, 0:3]
type_counts_inner_city = type_counts_inner.sum(axis=1)
exo_type_counts_wo_tourists = type_gb_counts[:, 3:6]


# Tourist population
gebied_tourist_population = (panel_covariates_demand[['gb', 'pop_tourists_total']]
                             .drop_duplicates()
                             .sort_values(by=['gb'], ascending=True)
                             ['pop_tourists_total'].to_numpy())

# Get the gebied-level panel covariates
gebied_level = panel_covariates_demand[["gb", "pop_tourists_total", "pop_tourists_hotels", "pop_tourists_airbnb"]].copy()
# Sort the gebieds by gb
gebied_level = gebied_level.sort_values(by=['gb'], ascending=True).drop_duplicates()
hotel_beds = panel_covariates_demand[['hotel_beds']].to_numpy()

# Get the number of hotel tourists in each gebied
gebied_tourist_population_hotels = gebied_level['pop_tourists_hotels']

exo_type_counts = np.c_[exo_type_counts_wo_tourists, gebied_tourist_population]

# Data on other controls
other_controls_mat = panel_covariates_demand[panel_covariates_demand['year'] == 2017][controls_vars]

# Housing stock
tenancy_counts = (panel_covariates_demand[['gb', 'tenancy_status_1', 'tenancy_status_2', 'log_area_by_usage_1']]
                  .drop_duplicates()
                  .sort_values(by=['gb'], ascending=True))

houses_LT = tenancy_counts['tenancy_status_1'].to_numpy()
gebied_housing_supply = pd.read_csv(filepath + f"inputs/str_ltr_gebied.csv")
houses_landlords = gebied_housing_supply['quantity_ltr'] + gebied_housing_supply['quantity_str']

# Average squared footage in locations
avg_squared_footage = np.exp(tenancy_counts['log_area_by_usage_1'].to_numpy())

#### Data on total incomes spent on consumption
amenity_file_df = pd.read_csv(filepath + f"inputs/panel_amenities_structural_estimation.csv")

#RESIDUALS OF AMNEITIES
amenity_file_df = amenity_file_df.loc[amenity_file_df['year'] == 2017]
amenity_file_df = amenity_file_df.loc[amenity_file_df['s'] == 1]
amenity_file_df = amenity_file_df.sort_values(by=['s', 'gb'], ascending=True)
exo_types_expenditure_consumption = amenity_file_df[[('budget_h' + str(k)) for k in range(4, 8)]].to_numpy()

#### Airbnb
tourist_demand_data_df = pd.read_csv(filepath + f"inputs/gebied_tourist_demand_covariates.csv")
tourist_demand_data_df = tourist_demand_data_df[tourist_demand_data_df['gb_code'] != 0]
tourist_demand_data_df = tourist_demand_data_df.sort_values(by=['gb_code'], ascending=True)
total_population = np.sum(tourist_demand_data_df['pop_tourist'])
tourist_demand_data_df = tourist_demand_data_df[:-1]




# Get observed Airbnb prices
p_observed = tourist_demand_data_df['price_' + config['agg_type']].to_numpy()
#OLD VERSIOMN
p_observed = (gebied_housing_supply['price_str'] / 365).to_numpy()


str_guests_to_total_guests = panel_covariates_demand['pop_tourists_airbnb'].to_numpy()[0:P['J']] / tourist_demand_data_df['pop_airbnb_commercial'].to_numpy()[0:P['J']]


# Population of tourists
guests_per_booking = tourist_demand_data_df['mean_accommodates'].to_numpy()

# Conversion factor
#conversion_factor = tourist_demand_data_df['conversion_to_units'].to_numpy()
conversion_factor = (airbnb_population / gebied_housing_supply['quantity_str']).to_numpy()

# Number of listings
n_listings = gebied_housing_supply['quantity_str'].to_numpy()

# Airbnb controls
tourist_demand_controls_data = tourist_demand_data_df['log_review_scores_location'].to_numpy()

#### Distance matrix
dist_mat = pd.read_csv(filepath + f"inputs/dist_mat_centroids.csv")

################################################################################
# Unpack and define parameters
################################################################################
# Number of groups, number of amenities, number of locations
K = 3
S = P['S']
J = P['J']

# Number of exogenous amenities
n_exo_amenities = 0

# Basic Parameters
P = {
    "K": K,
    "S": S,
    "J": J,
    "D": J+1,
    "tau_bar": 2,
    "Pop": Pop,
    "Pop_w_students": type_gb_counts,
    "y": 2017,
    "include_tau_dummy": True,
    "endo_airbnb": True,
    "include_airbnb_tourists": True,
    "touristic_amenities_only_central": False,
    "airbnb_tax_rate": 0.0,
    "n_exo_amenities": n_exo_amenities,
    "n_endo_amenities": S - n_exo_amenities,
    "id_exo_amenities": [],
    "id_endo_amenities": list(range(0, S - n_exo_amenities)),
    "theta_resident": np.array(theta_resident[:, 1:], dtype=float),
    "X_resident_parameters": np.array(X_resident_parameters, dtype=float)
}



# Utility parameters
demand_resid_x = (P["J"]+1)*P["tau_bar"]
demand_resid_y = P["J"]+1
demand_resid_z = P["K"]

Util_param = {
    "delta_r": rent_coefs[0:3],
    "delta_tau": tau_coef[0:3].flatten(),
    "delta_a": amenity_coefs[:, 0:3].T,
    "delta_covariates": controls_coefs_w_const[:, 0:3],
    "delta_j": loc_FEs[:, 0:3],
    "delta_t": time_FEs[9, 0:3],
    "Cobb_Douglas_housing": cobb_douglas_housing[0:3],
    "Cobb_Douglas_amenities": 1. - cobb_douglas_housing[0:3],
    "beta": 0.85,
    "w": income_endo_types[0:3],
    "demand_residuals": np.zeros(shape = (demand_resid_x, demand_resid_y, demand_resid_z)),
    "controls": other_controls_mat
}
P = merge(P, Util_param)

# Tourist demand coefficients
Tourist_param = {
    "delta_p_tourist": tourist_demand_price_coef,
    "delta_a_tourist": tourist_demand_amenity_coef,
    "delta_c_tourist": tourist_demand_controls,
    "str_guests_to_total_guests": str_guests_to_total_guests,
    "tourist_demand_controls": np.append(tourist_demand_controls_data[0:P["J"]], 0)
}
P = merge(Tourist_param, P)

# Moving cost parameters
moving_cost_param = {
    "gamma_0": gamma_0_coef[0:3].flatten(),
    "gamma_1": gamma_1_coef[0:3].flatten(),
    "gamma_2": gamma_2_coef[0:3].flatten(),
    "dist_mat": dist_mat
}
P = merge(P, moving_cost_param)

# Amenity parameters
Amenity_param = {
    "sigma_s": np.ones(shape = S),
    "alpha_ks": alpha_estimates,
    "gamma": gamma,
    "amenity_norm": amenity_norm,
    "w": income_endo_types[0:3],
    "income_exo_types": income_exo_types,
    "amenities": yearly_disp_income, #can be removed, called yearly_disp_income in julia
    "total_CD": 1. - cobb_douglas_housing, #7 elements in julia
    "exo_type_counts": exo_type_counts, #Can be removed
    "div_const": 1, # constant by which we divide wages to ease computation
    "lambda": 1,
    "exo_types_expenditure": exo_types_expenditure_consumption,
    "amenity_loc_FEs": location_FE_amenity_supply,
    "amenity_time_FE": time_FE_y_amenity_supply,
    "amenity_resid": amenity_supply_residuals.T,
    "exo_amenities": np.zeros(shape = (P["J"], 0))
}

P = merge(P,Amenity_param)

# Supply parameters
Supply_param = {
    "alpha": alpha_housing_supply[0],
    "p": p_observed,
    "total_tourist_population": total_population,#np.reshape(gebied_tourist_population, (P["J"], 1)).flatten(), 
    "pop_hotel_tourists": gebied_tourist_population_hotels.to_frame().to_numpy().flatten(),
    "disp_income_tourists": yearly_disp_income[-1],
    "cons_exp_share_tourists": all_cons_share_tourists,
    "hotel_beds": hotel_beds,
    "listings_to_total_str_guests": conversion_factor,
    "guests_per_booking": guests_per_booking,
    "owner_occupied": houses_LT,
    "H": houses_landlords.to_numpy().flatten(),
    "avg_squared_footage": avg_squared_footage
}

P = merge(P,Supply_param)

##################################################
# Implement counterfactuals
##################################################
P = merge(P, {"no_airbnb" : (config["counterfactual"] == "no_airbnb")})

Z = dict()
if config["counterfactual"] == "homogenous_preferences":
    Z['delta_tau'] = np.average(P['delta_tau'], weights = Pop) * np.ones(3)
    Z['delta_a'] = np.average(P['delta_a'], axis = 0, weights = Pop) * np.ones((3, P['S']))
    Z['delta_covariates'] = np.tile(np.average(P['delta_covariates'], axis = 1, weights = Pop), (3,1)).T
    Z['delta_j'] = np.tile(np.average(P['delta_j'], axis = 1, weights = Pop), (3, 1)).T
    Z['delta_t'] = np.average(P['delta_t'], weights = Pop) * np.ones(3)
    
    Z['Cobb_Douglas_housing'] = np.average(P['Cobb_Douglas_housing'], weights = Pop) * np.ones(3)
    Z['Cobb_Douglas_amenities'] = 1 - Z['Cobb_Douglas_housing']

    Z['gamma_0'] = np.average(P['gamma_0'], weights = Pop) * np.ones(3)
    Z['gamma_1'] = np.average(P['gamma_1'], weights = Pop) * np.ones(3)
    Z['gamma_2'] = np.average(P['gamma_2'], weights = Pop) * np.ones(3)

    Z['alpha_ks'] = np.concatenate([np.average(alpha_estimates[0:3,:], axis = 0, weights = Pop) * np.ones((3,6)), alpha_estimates[3:, :]])

P = merge(P,Z)

##################################################
# Implement matrices
##################################################
"""
    Constructs a tensor describing the evolution of location tenure for a given type.
    Entries are indexed by (tau, j, d, tau').
    Returns:
        A NumPy array representing the tensor describing the evolution of location tenure for the given type.
"""
def T_stochastic_k(P, tau_trans_probs, t, k):
    T_i_tensor = np.zeros((P['tau_bar'], P['J'] + 1, P['J'] + 1, P['tau_bar']), dtype=np.float64)
    for d in range(P['J']+1):
        for j in range(P['J']+1):
            if j == d:
                trans_prob_12 = tau_trans_probs[(tau_trans_probs.gb == j+1) &
                                                (tau_trans_probs.year == t) &
                                                (tau_trans_probs.combined_cluster == k+1)].transition_prob.values[0]

                T_i_tensor[:, j, d, :] = np.reshape([ (1 - trans_prob_12), trans_prob_12, 0, 1], (2,2))
            else:
                T_i_tensor[:, j, d, 0] = np.ones(P['tau_bar'])
    return T_i_tensor   

"""
    Creates a tensor for the stochastic evolution of location capital for all types
    Note about inputs: tau_trans_probs - DataFrame, a dataframe with estimated transition probabilities for location capital
"""
def T_stochastic_(P, tau_trans_probs, t):
    
    T_i_tensor_all_types = np.zeros((P['tau_bar'], P['J']+1, P['J']+1, P['tau_bar'], P['K']))
    for k in range(P['K']):
        T_i_tensor_all_types[:, :, :, :, k] = T_stochastic_k(P, tau_trans_probs, t, k)
    return T_i_tensor_all_types

Stochastic_Location_Capital_Matrix = T_stochastic_(P, tau_trans_probs, 2017)



m0_mat = np.kron(
              np.block([[np.ones((J, J)) - np.diag(np.ones(J)), np.zeros((J, 1))],
                          [np.zeros((1, J + 1))]]), np.ones(2)).T

res_matrix = np.hstack((P['dist_mat'], np.zeros((22, 1))))
res_matrix = np.vstack((res_matrix, np.zeros((1, 23))))
m1_mat = np.kron(res_matrix, np.ones(2)).T


first_part = np.zeros((J+1, J+1))
first_part[:, -1] = 1.0
first_part[-1, :] = 1.0
first_part[-1, -1] = 0.0
m2_mat = np.kron(first_part, np.ones(2)).T

Moving_Costs = np.zeros((46,23,3))
Moving_Costs[:, :, 0] = m0_mat
Moving_Costs[:, :, 1] = m1_mat
Moving_Costs[:, :, 2] = m2_mat


x = (tau_trans_probs.to_numpy()[:, 3]).reshape(69,1)
transprobs = np.zeros((23,3))
transprobs[:, 0] = x[0:23].flatten()
transprobs[:, 1] = x[23:46].flatten()
transprobs[:, 2] = x[46:69].flatten()



# Important matrices
Important_matrices = {"T_stochastic" : Stochastic_Location_Capital_Matrix,
                      "Moving_Costs_Matrix" : Moving_Costs,
                      'Transition_Probabilities': transprobs}
P = merge(P,Important_matrices)

P['tau_mat'] = np.zeros((46,23, 3))

for k in range(P['K']):
    J = P['J']
    transition_prob_ = jnp.array(P['Transition_Probabilities'])
    tau_mat =  jnp.vstack([transition_prob_[:, k], jnp.ones(J + 1)])
    undiag = P['delta_tau'][k] * tau_mat

    test = np.zeros((46,23))

    for i in range(23):
        test[2*i, i] = undiag[0, i]
        test[2*i+1, i] = undiag[1, i]

    P['tau_mat'][:, :, k] = test / P['delta_tau'][k]

P['tau_mat'] = jnp.array(P['tau_mat'])


################################################################################
# Load Pre-Estimated Calibrated Parameters
################################################################################
calibrated_deltas = pd.read_csv(filepath + f"inputs/gebied_tourist_demand_covariates.csv")['delta_j_gebied']
P["delta_j_tourist"] = calibrated_deltas[0:-1].to_numpy()

print(P["delta_j_tourist"])

#calibrated_deltas = pd.read_csv(filepath + f"estimates/delta_j.csv", header=None)
#P["delta_j_tourist"] = calibrated_deltas[:-1].to_numpy().flatten()
#print(P["delta_j_tourist"])

short_term_housing_shares = gebied_housing_supply

str_ = short_term_housing_shares['quantity_str']
ltr = short_term_housing_shares['quantity_ltr']

short_term_shares = np.divide(str_, (str_ + ltr))
p = P['p']

r = initial_r


housing_supply_data = short_term_housing_shares
ltr_share = housing_supply_data['quantity_ltr'] / (housing_supply_data['quantity_ltr'] + housing_supply_data['quantity_str'])
str_share = 1 - ltr_share
price_gap = housing_supply_data['price_ltr'] - housing_supply_data['price_str']

kappa = (np.log(ltr_share) - np.log(str_share) - P['alpha'] * price_gap / 10**4).to_numpy()

P["kappa"] = kappa

print("Successfuly Generated All Parameters!")
print("************************************************************")
print("************************************************************")
print("************************************************************")
print("Performing Sanity Checks")
print("************************************************************")

################################################################################
# Sanity checks
################################################################################

P['airbnb_extra_fee'] = 0
with open('config.json', 'r') as f:
    config = json.load(f)
P['no_airbnb'] = config['no_airbnb']
P['remove_students_from_supply'] = config['remove_students_from_supply']
P['old_version'] = config['old_version']
P['endogenous_tourist_choices'] = config['endogenous_tourist_choices']
P['amenity_tax_rate'] = 0

#P['amenity_norm'] = 10**7
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
_set_P(P)


#############################################

### Amenities
# Compute predicted amenities at the observed demand
r = initial_r
D = P['Pop_w_students'][:, 0:3]
a = initial_a
a_j = np.array(np.sum(initial_a, axis=1))


pred_amenities_observed = Amenity_supply_exo(a, P['p'], r, D, a_j)


# Check
difference = np.mean(np.subtract(pred_amenities_observed, initial_a), axis=0)
print(difference)
for i in range(S):
   print("Mean difference between the predicted amenity " + str(i) + " and the observed value:" + str(difference[i]))


u_hat_df = pd.read_csv(filepath + f"estimates/demand_location_choice_u_hat.csv")
u_hat_df = u_hat_df[u_hat_df["year"] == 2017]
u_hat_df = u_hat_df.sort_values(by=["combined_cluster", "gb"])
u_hat_vec = u_hat_df.iloc[:, 3:]
u_hat_vec = u_hat_vec.dropna().to_numpy()
u_hat = np.reshape(u_hat_vec, (P["J"], P["K"]), order = 'F')
u_hat = u_hat[:, :]

r = initial_r
D = type_counts_inner
a = initial_a


for k in range(P["K"]):
    _, util = static_utility(r, a, k)
    util = util.T[:, k][:-1]

    print("********************")
    print(f"Type {k}:")
    
    print("NEW Max difference in utility " + f"Type {k}:   " + str(np.linalg.norm(util - u_hat[:, k], np.inf)))
    print("NEW Mean difference in utility " + f"Type {k}:   " + str(np.mean(np.subtract(util, u_hat[:, k]), axis=0)))

print("************************************************************")
print("************************************************************")
print("************************************************************")
print("Sanity Checks Complete")
print("************************************************************")
print("************************************************************")
print("************************************************************")

##########################################################
# Dump the relevant variables to a Python-parseable format
##########################################################
# Set the JAX backend to X64
jax.config.update("jax_enable_x64", True)

# Create a Python dictionary to store the model parameters
p_json = {}

# Iterate over the model parameters and convert them to NumPy arrays
for k, v in P.items():
    if isinstance(v, np.ndarray):
        p_json[str(k)] = jnp.array(v)
    elif isinstance(v, pd.DataFrame):
        p_json[str(k)] = jnp.array(v.to_numpy())
    else:
        p_json[str(k)] = v

# Create the directory to store the model parameters
os.makedirs(f"Binaries/{experiment_name_counterfactual}", exist_ok=True)

# Save the model parameters to a NumPy file
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/initial_a.npy", jax.numpy.array(initial_a))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/initial_r.npy", jax.numpy.array(initial_r))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.year.npy", jax.numpy.array(tau_trans_probs['year']))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.gb.npy", jax.numpy.array(tau_trans_probs['gb']))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.combined_cluster.npy", jax.numpy.array(tau_trans_probs['combined_cluster']))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.transition_prob.npy", jax.numpy.array(tau_trans_probs['transition_prob']))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.total_decision.npy", jax.numpy.array(tau_trans_probs['total_decision']))
jax.numpy.save(f"Binaries/{experiment_name_counterfactual}/tau_trans_probs.total_by_tau.npy", jax.numpy.array(tau_trans_probs['total_by_tau']))
jnp.save(f"Binaries/{experiment_name_counterfactual}/P.npy", p_json)

np.savetxt("../../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_a.csv", initial_a, delimiter=",")
np.savetxt("../../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_r.csv", initial_r, delimiter=",")
np.savetxt("../../../data/simulation_results/"+gamma_string+"/equilibrium_objects/initial_p.csv", np.array(P['p']), delimiter=",")

print("Successfuly Saved All Binaries!")
print("Binaries Located At Binaries/" + str(experiment_name_counterfactual))
print("Modify config.json as needed, and proceed to solving model")
print("************************************************************")
print("************************************************************")
print("************************************************************")
