###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

##################################################################################################################################
# This file contains the main functions used in the economic model as described in the paper. See paper for more details.
##################################################################################################################################

# Required library imports
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jaxopt import AndersonAcceleration
import numpy as np

# Global Variables
global P
global year
global gb
global combined_cluster
global transition_prob

def _set_P(tup):
    global P
    P = tup

# Sets all global constants as needed
def _set_constants(constant_tup):
    global P
    global year
    global gb
    global combined_cluster
    global transition_prob
    P, year, gb, combined_cluster, transition_prob = constant_tup

# Constructs a DEV matrix for stochastic transitions of location capital for a given type
def DEV_stochastic(EV, k):
    EV_mat = jnp.reshape(EV, (P['tau_bar'], P['J']+1), order='F')
    EV_mat_rep = jnp.repeat(EV_mat.T, P['tau_bar']*(P['J']+1), axis=0)
    T_mat = jnp.reshape(P['T_stochastic'][:,:,:,:,k], ((P['J']+1)*P['tau_bar']*P['D'], P['tau_bar']),  order='F')
    DEV_mat_full = T_mat * EV_mat_rep
    DEV_mat_avg = DEV_mat_full @ jnp.ones(P['tau_bar'])
    DEV_mat = jnp.reshape(DEV_mat_avg, ((P['J']+1)*P['tau_bar'], P['D']), order='F')
    return DEV_mat


# Constructs a tensor describing the evolution of location tenure for a give type. Entries are indexed by (τ,j,τ',d).
# Note about inputs: tau_trans_probs - DataFrame, a dataframe with estimated transition probabilities for location capital
def T_stochastic_k_transitions(year, gb, combined_cluster, transition_prob, t, k):
    T_i_tensor = jnp.zeros((P['tau_bar'], P['J']+1, P['tau_bar'], P['D']))

    for d in range(P['D']):
        for j in range(P['J']+1):
            if j == d:
                # Find the index in arrays that matches the conditions
                trans_prob_12 = jnp.where((gb == j) & (year == t) & (combined_cluster == (k+1)), x = transition_prob, y = jnp.zeros_like(gb))
                trans_prob_12 = jnp.sum(trans_prob_12)
                T_i_tensor = T_i_tensor.at[:, j, :, d].set(jnp.array([[1-trans_prob_12, trans_prob_12], [0, 1]]))
            else:
                T_i_tensor = T_i_tensor.at[:, j, 0, d].set(jnp.ones(P['tau_bar']))
                
    return T_i_tensor

# Computes the stationary distribution of a Markov Chain given its transition matrix
@jax.jit
def stationary_dist_MC(M):
    n = M.shape[0]
    Q = M.T - jnp.eye(n)
    Q = jnp.vstack([Q[:-1], jnp.ones(n)])
    b = jnp.zeros(n)
    b = b.at[-1].set(1.0)
    return solve(Q, b)


# static utility calculation, no amenitites. Computes the flow utility of type k for vector of prices r and matrix of amenities a
@jax.jit
def static_utility(r, a, k):
    x_resident_exo = jnp.copy(jnp.log(r))
    x_resident_exo = x_resident_exo.reshape(-1, 1)
    x_resident_exo = jnp.hstack((x_resident_exo, jnp.log(a)))
    x_resident_exo = jnp.hstack((x_resident_exo, P['non_amentity_params']))
    utility = jnp.matmul(x_resident_exo, P["theta_resident"]).T
    utility = jnp.hstack((utility, jnp.zeros((3, 1))))
    utility = jnp.reshape(utility, (3, 23))
    res_utility = utility[k, :]
    res_utility = jnp.reshape(res_utility, (1, 23))
    res = jnp.repeat(res_utility, 46, axis=0).reshape(46, -1)
    
    return res, utility

@jax.jit
def utility(r,a,k):
    u_flow, res =  static_utility(r,a,k)

    if P['static']:
        return u_flow

    moving_costs_0 = P['Moving_Costs_Matrix'][:, :, 0]
    moving_costs_1 = P['Moving_Costs_Matrix'][:, :, 1]
    moving_costs_2 = P['Moving_Costs_Matrix'][:, :, 2]

    tau_mat = P['tau_mat'][:, :, k]

    res_new = u_flow + moving_costs_0*P["gamma_0"][k] + moving_costs_1 * P["gamma_1"][k] + moving_costs_2*P["gamma_2"][k] + tau_mat*P['delta_tau'][k]
    return res_new



# Computes the EV function and a transition matrix for specified rental prices, amenities, and a household type. Allows the user to supply
# custom initial guess for the EV vector. Note that we always index the rows of objects by (j,τ) (e.g. the flow utility matrix is
# indexed by [(j,τ),d]).
# Output:
#   - Pi_mat - a τ_bar x J square matrix giving the transition probabilities, states are indexed by (j,τ)
#    - EV_G - a τ_bar x J vector giving the values of EV function. Entries are indexed by (j,τ)
@jax.jit
def trans_mat_EV_mat_exo(r, a, k, EV_initial):
    #### Compute flow utility
    flow_U = utility(r, a, k)

    # Initialize variables
    dist_func = 1.0
    i = 0

    # iterate
    val= (i, dist_func, EV_initial)
    
    def cond_fun(val):
        i, dist_func, _ = val
        return (dist_func > P['tol']) & (i <= P['max_iter'])

    def body_fun(val):
        i, _, EV_G = val
        # Prepare the inclusive value
        if P['static'] or P['myopic']:
            v = flow_U
        else:
            DEVc_mat = DEV_stochastic(EV_G,k)

            v = flow_U + P['beta']*DEVc_mat

        # Overfloating adjustment
        v_max = jnp.max(v)

        # Update guess
        EV = v_max + jnp.log(jnp.sum(jnp.exp(v - v_max), axis=1))
        
        # Compute convergence criterion
        dist_func = jnp.linalg.norm(EV-EV_G,jnp.inf)

        # Update vars
        i += 1
        return (i, dist_func, EV)

    _, _, EV_G = jax.lax.while_loop(cond_fun, body_fun, val)

    #### Compute the transition matrix
    # Prepare a matrix of choice probabilities
    if P['static'] or P['myopic']:
        v = flow_U 
    else:
        DEVc_mat = DEV_stochastic(EV_G,k)
        v = flow_U + P['beta']*DEVc_mat

    v_max = jnp.max(v)
    v = v - v_max
    denom = jnp.sum(jnp.exp(v), axis=1)
    Q_mat = jnp.exp(v) / jnp.tile(denom, (P['J']+1,1)).T

    # Get the transition matrix for location capital    
    if not P['static']:
        T_tensor = T_stochastic_k_transitions(year, gb, combined_cluster, transition_prob, P['y'], k)
        T_mat = jnp.reshape(T_tensor,(P['tau_bar']*(P['J']+1),P['D']*P['tau_bar']), order='F')
        # Compute an auxiliary transition matrix
        Pi_mat = T_mat * jnp.kron(Q_mat,jnp.ones((1,2)))
    else:
        Pi_mat = jnp.kron(Q_mat,jnp.ones((1,2)))
    
    # Return the EV function and the transition matrix
    return Pi_mat, EV_G


# Returns the fraction of long-term houses for each location
@jax.jit
def S_L(r: jnp.array, p: jnp.array): #Convert this to quantities
    # In the no airbnb counterfactual, we have 0 short-term rentals
    if P["no_airbnb"]:
        return jnp.ones_like(r)
    
    # Compute the corresponding terms (note that the supply regression was performed with price unit of 10,000 euros)
    # We need to adjust the price of long-term rental, which is expressed in meters
    # To do so, we use the average squared footage in locations
    r = r * P['avg_squared_footage'] / 1e4

    # Airbnb tax rate
    p_annual = (1 - P['airbnb_tax_rate']) * (p * 365) / 1e4 # airbnb_tax_rate is deprecated

    E_supply = jnp.exp(P['alpha'] * r - P['alpha'] * (p_annual) + P['kappa'])
    prob_supply = jnp.ones(P['J']) / (E_supply + 1)
    return 1 - prob_supply



@jax.jit
def tourist_guest_demand(p, a):
    p = p + P['airbnb_extra_fee']

    U = P['delta_j_tourist'] + P['delta_p_tourist'] * jnp.log(jnp.where(p > 0, p, 1) / P['guests_per_booking']) + jnp.matmul(jnp.log(jnp.where(a > 0, a, 1)), P['delta_a_tourist']) + P['delta_c_tourist'] * P['tourist_demand_controls'][:P['J']]

    # indirect utility -- price entered should be price per night
    U = jnp.concatenate([U, jnp.array([0])], axis=0)

    U_norm = U - U[-2]

    # Compute demand with type I EV errors (from one tourist)
    exp_U = jnp.exp(U_norm)
    D_one_tourist = exp_U / jnp.sum(exp_U)  

    return D_one_tourist * jnp.sum(P["total_tourist_population"])


def _tourist_guest_demand(p,a):
    theta_tourist = np.reshape(np.hstack([P['delta_p_tourist'], P['delta_a_tourist'], P['delta_c_tourist']]), (8,1))
    p_guest = np.hstack([jnp.log(jnp.where(p > 0, p, 1) / P['guests_per_booking']), (0)])
    a_guest = np.vstack([jnp.log(jnp.where(a > 0, a, 1)), jnp.zeros(6)])
    p_guest = np.reshape(p_guest, (23,1))
    P['tourist_demand_controls'] = np.reshape(P['tourist_demand_controls'], (23,1))
    
    inter = np.hstack ([p_guest, a_guest, P['tourist_demand_controls']])

    U = np.matmul(inter, theta_tourist) + np.reshape(np.hstack([P['delta_j_tourist'], 0]), (23,1))

    U_norm = U - U[-2]
    # Compute demand with type I EV errors (from one tourist)
    exp_U = jnp.exp(U_norm)
    D_one_tourist = exp_U / jnp.sum(exp_U)  

    return (D_one_tourist * jnp.sum(P["total_tourist_population"])).flatten()


# Computes demand from tourists for Airbnb listings in different neighborhoods.
@jax.jit
def D_S(r, p, a):
    # mini-experiment for taxation: charge a flat fee per night
    if P["no_airbnb"]:
        return jnp.zeros_like(r)
    else:
        if P['endogenous_tourist_choices']:
            return tourist_guest_demand(p, a) * jnp.append(P['str_guests_to_total_guests'], 1)
        else:
            Non_Conv = (1 - S_L(r, p)) * P['H']
            S_airbnb = Non_Conv * P['listings_to_total_str_guests']
            return S_airbnb
    # Return the total demand
    # total num of tourists including the hotels and outside the gebied




# Computes the excess demand function for short-term rentals
@jax.jit
def ED_S(r, p, a):
    if P["no_airbnb"]:
        return jnp.zeros_like(p)
    
    # Compute demand for short-term rentals
    # Notice we discard the J+1th option, which signifies staying in a hotel,
    # as we discarded the out-of-city option for long-term housing
    D_airbnb = D_S(r, p, a)[:P['J']] / P['listings_to_total_str_guests']

    # Compute supply of short-term rentals
    S_airbnb = (1 - S_L(r, p)) * P['H']

    # Compute the excess demand for short-term rentals
    return jnp.divide(D_airbnb - S_airbnb, .5*(D_airbnb + S_airbnb)) * P['include_airbnb_tourists']





#For a vector of rental prices and amenities, computes the demand for long-term housing based on the stationary distribution
# Output:
#   - D_Ls - J x K matrix yielding the total demand of households of type k for long-term housing in location j
#   - EVs - (τ_bar*J) x K matrix of EV function values for each type
@jax.jit
def D_L(r, a, EVs_initial):
    def body_fn(var):
        k = var[0].astype(int)
        EVs_initial = var[1:]
        trans_mat, EV_k = trans_mat_EV_mat_exo(r, a, k, EVs_initial)

        stat_dist_k = stationary_dist_MC(trans_mat)
        stat_dist_all_k = jnp.reshape(stat_dist_k, (P['tau_bar'], P['J']+1), order='F')
        D_Ls_k = P['Pop'][k] * jnp.sum(stat_dist_all_k, axis=0).T

        return D_Ls_k, EV_k

    D_Ls, EVs = jax.lax.map(
    body_fn,
    jnp.concatenate([jnp.arange(P['K']).reshape(P['K'],1), EVs_initial.T], axis=1)
                        )
    return D_Ls.T, EVs.T





# Computes the demand for living area, conditional on living in the given location, returning a J x K matrix
@jax.jit
def D_sq_area(r):#
    return jnp.repeat((P['Cobb_Douglas_housing'] * P['w'])[jnp.newaxis, :], P['J'], axis=0) / r[:, jnp.newaxis]




# Computes the student rental quantity
@jax.jit
def student_SL():
    if not P['remove_students_from_supply']:
        return 0
    if(P['old_version']):
        return 0
    
    student_expenditure_shares = 1 - P["total_CD"][3]
    student_population = P["Pop_w_students"][:, 3].T
    student_income = P["income_exo_types"][0]
    rent_meter = P["rent_meter"] 
    student_housing_quantity = student_population * (student_expenditure_shares * student_income / rent_meter)
    return student_housing_quantity




# For a vector of rental prices and amenities and given stationary matrix of location choices, 
# computes the excess demand for long-term housing based on the stationary distribution amd demand for living area
# Output:
#   - Stat_ED_vec - J x K matrix yielding the excess demand of households of type k for long-term housing in location j
@jax.jit
def ED_L(r, p, DL_mat):
    # Get demand for living area conditional on living in a location
    demand_sq_area = D_sq_area(r)

    # Compute the total demand
    DL_mat = DL_mat * demand_sq_area

    # Aggregate across households by summing up to get the total demand
    D = DL_mat.sum(axis=1)

    # Multiply the housing supply by the average squared footage in the area to get total squared footage available in the location
    Supply = ((S_L(r, p) * P['H'] + P['owner_occupied']) * P['avg_squared_footage'] - student_SL())

    #Normalize this by dividing by Supply. Same for ED_S
    return jnp.divide(D - Supply, .5*(Supply + D))






@jax.jit
def calc_hotel_pop(r, p, a, P):
    total_hotel_bed = jnp.sum(P['hotel_beds'])
    pop_tourists_hotels = jnp.where(
        P['no_airbnb'],
        jnp.sum(P["total_tourist_population"]),
        D_S(r, p,a)[-1]
    )
    occupancy_rate = jnp.divide(pop_tourists_hotels, total_hotel_bed)
    return jnp.where(
        P['endogenous_tourist_choices'],
        jnp.multiply(occupancy_rate, P['hotel_beds']).flatten(),
        P['pop_hotel_tourists']
    )
    


# Computes supply of amenities in equilibrium, for a given vector of rental prices and demand in the steady state, for a model with homothetic preferences.
@jax.jit
def Amenity_supply_exo(a, p, r, D, a_j):
    """
    Note that this function has only been tested w/ zero exogenous amenities and endo_airbnb=True.
    """
    if P['exo_amenities']:
        return a

    # Compute expenditure shares
    budget = jnp.repeat(P['w'][jnp.newaxis, :], P['J'], axis=0) * jnp.repeat(P['Cobb_Douglas_amenities'][:P['K']][jnp.newaxis, :], P['J'], axis=0)
    total_budget_inner = (P['lambda'] * D * budget)

    pop_airbnb_w_outside = D_S(r, p, a)


    pop_airbnb = pop_airbnb_w_outside[:P['J']]

    total_hotel_bed = jnp.sum(P['hotel_beds'])
    pop_tourists_hotels = jnp.where(
        P['no_airbnb'],
        jnp.sum(P["total_tourist_population"]),
        pop_airbnb_w_outside[-1]
    )

    occupancy_rate = jnp.divide(pop_tourists_hotels, total_hotel_bed)
    pop_hotels = jnp.where(
        P['endogenous_tourist_choices'],
        jnp.multiply(occupancy_rate, P['hotel_beds']).flatten(),
        P['pop_hotel_tourists']
    )

    total_pop_tourists = pop_hotels + pop_airbnb

    total_expenditure_tourists = jnp.reshape(total_pop_tourists, (22,)) * (P['disp_income_tourists'] * jnp.ones(P['J']))
    exp_share = jnp.concatenate([total_budget_inner, P['exo_types_expenditure'][:,:-1], total_expenditure_tourists.reshape(total_expenditure_tourists.shape[0], 1) * P['cons_exp_share_tourists']], axis=1) @ P['alpha_ks']

    # Compute amenities equilibrium quantities
    Log_Amenity_supply_endo = jnp.log(jnp.divide(exp_share, P.get('amenity_norm', 10**7))) + (P['amenity_time_FE'] + P['amenity_loc_FEs'] + P['gamma']*jnp.log(a_j))[:, jnp.newaxis] + P['amenity_resid'] - np.log(1 + P['amenity_tax_rate'])
    Amenity_supply_endo = jnp.exp(Log_Amenity_supply_endo)

    return Amenity_supply_endo







# For a given matrix of amenities and an initial guess of equilibrium prices, computes equilibrium in the housing market
# using the tatonnement adjustment algorithm in an adaptive way. The update parameter multiplying the excess demand vector
# shrinks whenever the objective function increases between steps. When the update parameter gets too small as specified by
# the user, the function proceeds to employing the Nelder-Mead algorithm, using the last recorded vector of prices from the
# tatonnement algorithm as the initial value.
# Inputs (selected):
#    - δ - initial value for the output parameter on the ED vector in the tatonnement procedure
#    - shrinkage - a multiplicative factor which  determines the new value of the output parameter the value of the objective
#        function increases between steps
#    - update_param_tol - tolerance on δ. When the value of the update parameter goes below this value, the tatonnement algorithm
#        is terminated and we proceed with using the Nelder-Mead algorithm
#
# Outputs:
#    - x_g - equilibrium price vector
#    - δ_c - last value of the update parameter in the Tatonnement algorithm before it was terminated
@jax.jit
def tatonnement_prices(
        initial_r,
        initial_p,
        a, 
        max_iter, 
        EVs_initial, 
        delta, 
        shrinkage, 
        update_param_tol, 
        algo_tol, 
        tatonnement_direct
        ):

    DL, EVs_g = D_L(initial_r, a, EVs_initial)
    DL = DL[:P['J'],:] 
    EDL = ED_L(initial_r, initial_p, DL)
    EDS = jnp.where(
        P['old_version'],
        0,
        ED_S(initial_r, initial_p, a)
    )
        
    
    val = (
        1000, # tol
        0, # iteration counter
        delta, # delta_c
        initial_r, # x_g
        initial_p, # p_g
        EDL,
        EDS,
        EVs_g
    )
    
    def cond_fun(val):
        tol, i, delta_c, _, _, _, _, _ = val
        return (tol > algo_tol) & (i < max_iter) & (delta_c > update_param_tol)
    
    def body_fun(val):
        prev_tol, i, delta_c, x_g, p_g, EDL, EDS, EVs_g = val



        EDL_adjusted = jnp.maximum(delta_c * EDL, -delta_c * jnp.ones(P['J'])) * (EDL < 0) + jnp.minimum(delta_c * EDL, delta_c * jnp.ones(P['J'])) * (EDL > 0)
        x_g = x_g = jnp.where(
            P["constant_rent"],
            x_g,
            jnp.exp(jnp.log(x_g) + EDL_adjusted)
        )

        EDS_Adjusted = jnp.maximum(delta_c * EDS, -delta_c * jnp.ones(P['J'])) * (EDS < 0) + jnp.minimum(delta_c * EDS, delta_c * jnp.ones(P['J'])) * (EDS > 0)
        p_g = jnp.where(
            P["no_airbnb"],
            jnp.ones_like(p_g) * jnp.inf,
            jnp.where(
                P["airbnb_prices_exogenous"],
                p_g,
                jnp.exp(jnp.log(p_g) + EDS_Adjusted)
            )
        )
        
        DL,EVs_g = D_L(x_g,a, EVs_g)
        DL = DL[:P['J'],:] 
        EDL = ED_L(x_g, p_g, DL)
        EDS = jnp.where(
            P['old_version'],
            0,
            ED_S(x_g, p_g, a)
        )
        tol = jnp.maximum(
            jnp.linalg.norm(EDS, ord=jnp.inf),
            jnp.linalg.norm(EDL, ord=jnp.inf),
        )

        delta_c = jnp.where(tol >= prev_tol, shrinkage * delta_c, delta_c)

        return (tol, i+1, delta_c, x_g, p_g, EDL, EDS, EVs_g)

    tol, i, delta_c, x_g, p_g, _, _, EVs_g = jax.lax.while_loop(cond_fun, body_fun, val)

    return x_g, p_g, delta_c, tol, i, EVs_g


# Main function for the nested fixed-point algorithm for equilibrium computation. The function allows user to specify the solution algorithm
# in the inner loop.
# Inputs (selected):
#    - λ - coefficient in the convex combination of the previous guess of an amenity matrix and the the equilibrium amenity matrix from
#        the inner loop. λ multiplies the previous guess
#    - δ - initial value of the update parameter in the Tatonnement algorithm, if selected
#    - algo_tol - convergence tolerance on the algorithm in the inner loop
#    - tol - convergence tolerance on the algorithm in the outer loop
#    - tatonnement_start_over - if true, each inner loop iteration of the Tatonnement algorithm starts with the value of
#        the update parameter of δ; otherwise, it will take as the initial value the last value of δ from the last call of the inner loop
# Outputs:
#    - r_e - equilibrium vector of rental prices
#    - a_e - equilibrium matrix of amenities
@jax.jit
def outer_loop(
        initial_a,
        initial_r,
        initial_p,
        max_iter,
        lamb,
        delta,
        algo_tol,
        shrinkage,
        update_param_tol,
        max_iter_algo,
        tol,
        tatonnement_direct,
        EVs_initial,
        total_cost,
        a_g
    ):        
    
    val = (
        total_cost, # dist_a
        0, # number of iterations (i)
        a_g, # a_g
        initial_a, # a_e
        initial_r, # r_e
        initial_p, # p_e
        EVs_initial, # EVs_g
        (0.0, 0.0, 0)
    )

    def cond_fun(val):
        total_cost, i, _, _, _, _, _, _ = val
        return (i < max_iter) & (total_cost > tol)


    def body_fun(val):
        _, i_outer, a_g, a_e, r_e, p_e, EVs_g, _ = val
        
        r_e, p_e, delta_c, tol, i_tatonnement, EVs_g = tatonnement_prices(
            r_e,
            p_e,
            a_g,
            max_iter_algo,
            EVs_g,
            delta,
            shrinkage,
            update_param_tol,
            algo_tol,
            tatonnement_direct
        )


        tatonnement_logs = (delta_c, tol, i_tatonnement)
        
        DL, EVs_g = D_L(r_e, a_g, EVs_g)
        DL = DL[:P['J'], :]

        a_e = Amenity_supply_exo(a_g, p_e, r_e, DL, jnp.sum(a_g, axis=1))

        a_g_new  = (1- lamb) * a_e + lamb * a_g
        dist_a = jnp.linalg.norm(a_g - a_g_new, ord=jnp.inf)

        old_entry_cost = P['gamma']*jnp.log(jnp.sum((a_g), axis= 1).T)
        new_entry_cost = P['gamma']*jnp.log(jnp.sum((a_g_new), axis= 1).T)

        previous_amenities_w_entry = jnp.append(a_g, jnp.reshape(old_entry_cost, (22,1)), axis = 1)
        new_amenities_w_entry = jnp.append(a_g_new, jnp.reshape(new_entry_cost, (22,1)), axis = 1)

        # New Equilibrium Condition
        if P['amenity_entry_costs_equilibrium_condition']:
            total_cost = jnp.linalg.norm(previous_amenities_w_entry - new_amenities_w_entry, ord=jnp.inf)
        else:
            total_cost = dist_a
            

        return (total_cost, i_outer+1, a_g_new, a_e, r_e, p_e, EVs_g, tatonnement_logs)

    total_cost, i_outer, a_g, a_e, r_e, p_e, EVs_g, tatonnement_logs = jax.lax.while_loop(cond_fun, body_fun, val)

    jax.debug.print("Tolerance: {}", total_cost)

    return r_e, p_e, a_e, EVs_g, total_cost, a_g, i_outer, tatonnement_logs

"""
Below are auxilliary functions that compute welfare statistics etc.
"""
@jax.jit
def entropy_index_locations(D):
    D_j = jnp.sum(D, axis=1)
    d = D / D_j[:, jnp.newaxis]
    return - jnp.sum(d * jnp.log(d), axis=1)

@jax.jit
def entropy_index_city(D):
    D_k = jnp.sum(D, axis=0)
    D_j = jnp.sum(D, axis=1)
    Dsum = jnp.sum(D)
    H_hat = - jnp.sum((D_k/Dsum) * jnp.log(D_k/Dsum))
    H_bar = jnp.sum(entropy_index_locations(D) * (D_j/Dsum))
    return (H_hat - H_bar) / H_hat

def stat_dist_exo(r, a, EVs_initial):
    def body_fn(var):
        k = var[0].astype(int)
        EVs_initial = var[1:]
        trans_mat, EV_k = trans_mat_EV_mat_exo(r, a, k, EVs_initial)
        stat_dist_k = stationary_dist_MC(trans_mat)
        stat_dist_all_k = jnp.reshape(stat_dist_k, (P['tau_bar'], P['J']+1), order='F')
        return stat_dist_all_k, EV_k

    stat_dist, EVs = jax.lax.map(
    body_fn,
    jnp.concatenate([jnp.arange(P['K']).reshape(P['K'],1), EVs_initial.T], axis=1)
                        )
    return stat_dist.T, EVs.T

def consumer_surplus(r, a, EVs_initial):
    stat_dist_all, EVs = stat_dist_exo(r, a, EVs_initial)
    stat_dist_all_mat = jnp.reshape(stat_dist_all, ((P['J'] + 1) * P['tau_bar'], P['K']))
    W = jnp.sum(EVs * stat_dist_all_mat, axis=0)
    delta_r = P['delta_r']
    exp_shares = 1 - P['Cobb_Douglas_housing']
    muc = -1 * np.multiply(1/(1 - P['beta']), (delta_r / (np.multiply(P['w'], exp_shares))))
    return np.divide(W, muc)




def welfare_households(r, a, EVs_initial):
    EVs_initial = jnp.ones((int(P["tau_bar"])*(int(P["J"])+1),int(P["K"])))
    stat_dist_all, EVs = stat_dist_exo(r, a, EVs_initial)
    stat_dist_all_mat = jnp.reshape(stat_dist_all, ((P['J'] + 1) * P['tau_bar'], P['K']))
    W = jnp.sum(EVs * stat_dist_all_mat, axis=0)
    return W

def welfare_landlords(r, P):
    """
    Compute the welfare of landlords at rental prices r.
    """
    # Update rent to rent for the whole unit
    r_adjusted = r * P['avg_squared_footage']

    p_annual = (1 - P['airbnb_tax_rate']) * (P['p'] * 365)

    # Compute landlord welfare (replace placeholders with actual calculations)
    welfare_locwise = jnp.log(jnp.exp(P['alpha'] * r_adjusted / 10e4) + jnp.exp(P['alpha'] * p_annual / 10e4 - P['kappa']))

    return welfare_locwise



from scipy.optimize import brentq  # Assuming brentq for find_zero
def rent_equivalent(orig_welfare, r_eq, a_eq, P):
    """
    Compute the rent equivalent for each type of household.
    """
    max_bounds=10000*np.ones(P['K'])
    min_bounds=1*np.ones(P['K'])
    rent_equivs = np.zeros(P['K'])

    for k in range(P['K']):

        # Define the function to find the root
        def f(x):
            return welfare_households(r_eq + x * np.ones(P['J']), a_eq, P)[k] - orig_welfare[k]

        # Use Brentq method to find the root
        rent_equivs[k] = brentq(f, min_bounds[k] * min(r_eq) + 1e-6, max_bounds[k] * max(r_eq))

    return rent_equivs


def consumption_equivalent(r_orig, a_orig, r_eq, a_eq, rental_income_orig, rental_income_eq, P, delta_r, EVs_initial):
    """
    Compute the consumption equivalent for different types of households.
    """

    # Get welfare values
    W_0 = welfare_households(r_orig, a_orig, EVs_initial)
    W_1 = welfare_households(r_eq, a_eq, EVs_initial)

    beta = P['beta']
    exp_shares = 1 - P['Cobb_Douglas_housing']
    income = P['w']

    # Calculate consumption equivalent
    CE = np.multiply( np.exp( np.multiply((1 - beta) * np.abs((1 - exp_shares) / delta_r), (W_0 - W_1))), (income + rental_income_orig)) - income - rental_income_eq

    return CE

"""
function CE_renter(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P,δ_r)
    Computes the consumption equivalent using the closed-form solution from the paper for renters 
"""
def CE_renter(r_orig, a_orig, r_eq, a_eq, P, delta_r, EVs_initial):
    W_0 =  welfare_households(r_orig,a_orig,EVs_initial)
    W_1 =  welfare_households(r_eq,a_eq,EVs_initial)
    exp_shares = 1 - P['Cobb_Douglas_housing']
    muc = -1 * np.multiply(1/(1 - P['beta']), (delta_r / (np.multiply(P['w'], exp_shares))))
    CE =  np.divide((W_1-W_0), muc)
    return CE

"""
function CE_homeowner(r_orig::Vector{Float64},a_orig::Matrix{Float64},p_orig::Vector{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},p_eq::Vector{Float64},P,δ_r)
    Computes the consumption equivalent using the closed-form solution from the paper for homeowners 
"""
def CE_homeowner(r_orig, a_orig, r_eq, a_eq, P, delta_r, EVs_initial):
    
    # Welfare of renters
    CE = CE_renter(r_orig,a_orig,r_eq,a_eq,P,delta_r, EVs_initial) 

    # Income from rents
    rental_income_orig = np.average((np.multiply(P['avg_squared_footage'], r_orig.flatten())), weights = (P['H']) / (P['H']).sum())
    rental_income_eq = np.average((np.multiply(P['avg_squared_footage'], r_eq.flatten())), weights = (P['H']) / (P['H']).sum())

    CS_homeowner = CE + (rental_income_eq-rental_income_orig) * np.ones(P['K'])

    return CS_homeowner



def gini_coeff(data):
    """
    Computes the Gini coefficient for any allocation vector.
    """

    # Check for non-negative values
    if not all(x >= 0 for x in data):
        raise ValueError("All values in data must be non-negative.")

    # Sort the data
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Calculate Gini coefficient using the given formula
    gini = 2 * np.sum((np.arange(1, n + 1) * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n

    return gini


def gini_weighted(datadistarray):
    """
    Computes a weighted Gini coefficient.
    """

    # Check for non-negative weights
    if np.any(datadistarray[:, 1] < 0):
        print("Error: There are negative weight values. Use a different inequality measure.")
        return

    # Check for valid dimensions
    if datadistarray.shape[1] not in (1, 2):
        print("Error: Input data does not have conformable dimensions.")
        return

    # Handle case with weights only (one column)
    if datadistarray.shape[1] == 1:
        n = len(datadistarray)
        sorted_data = np.sort(datadistarray[:, 0])
        gini = 2 * np.sum((np.arange(1, n + 1) * sorted_data)) / (n * np.sum(sorted_data)) - 1

    # Handle case with data and weights (two columns)
    elif datadistarray.shape[1] == 2:
        swages = np.cumsum(datadistarray[:, 1] * datadistarray[:, 0])
        gweights = swages[1] * datadistarray[1, 0] + np.sum(datadistarray[2:, 0] * (swages[2:] + swages[1:-1]))
        gini = 1 - gweights / swages[-1]

    return gini
