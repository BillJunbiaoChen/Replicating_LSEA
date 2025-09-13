####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################

println("################################################################################")
println("####################### Initializing CF construction ###########################")
println()

# Set path
path = @__DIR__
cd(path)
cd("../../..")
main_path = pwd()

# Packages
using Random, LinearAlgebra, DataFrames, CSV, Distributions, StatsBase, GLM, Plots, Printf

# Functions
include("read_inputs.jl")

################################################################################
# Unpack and define parameters
################################################################################

# Number of groups, number of amenities, number of locations
#### Basic dimensions
P = (K = 3,
     S = 6,
     J = 22,
     D = 23,
     tau_bar = 2,
     include_airbnb = true,
     endogenous_tourist_choices = true) 

# Utility parameters
Util_param = (θ_resident,
              X_resident_exo = X_resident_exo,
              X_full = X_full,
              demand_residuals = zeros((P.J+1)*P.tau_bar,P.J+1,P.K),
              beta = beta)

P = merge(P, Util_param)

# Tourist demand coefficients
Tourist_param = (θ_tourist = θ_tourist,
                tourist_demand_controls = tourist_demand_controls_data,
                mean_accommodates = mean_accommodates,
                total_tourists = total_tourists,
                str_guests_to_total_guests = str_guests_to_total_guests,
                listings_to_total_str_guests = listings_to_total_str_guests,
                hotel_beds = hotel_beds)
P = merge(P, Tourist_param)

# Moving cost parameters
moving_cost_param = (gamma_0 = gamma_0_coef[1:3],
                     gamma_1 = gamma_1_coef[1:3],
                     gamma_2 = gamma_2_coef[1:3],
                     delta_tau = tau_coef[1:3],
                     dist_mat = dist_mat)
P = merge(P, moving_cost_param)

# Amenity parameters
Amenity_param = (alpha_ks = alpha_estimates[:,1:P.S],
                gamma = gamma,
                income = yearly_disp_income,
                pop_exogeneous = exo_type_counts,
                exp_shares = exp_shares,
                amenity_loc_FEs = location_FE_amenity_supply,
                amenity_time_FE = time_FE_y_amenity_supply,
                amenity_resid = amenity_supply_residuals)
P = merge(P,Amenity_param)

# Supply parameters
Supply_param = (alpha = alpha_housing_supply,
                H = houses_landlords,
                H_LT = houses_LT,
                avg_squared_footage = avg_squared_footage)
P = merge(P,Supply_param)

# Population 
Pop_param = (pop_exogeneous = exo_type_counts,
             Pop = type_counts[1:3],
             pop_hotel_tourists = gebied_tourist_population_hotels
             )
P = merge(P,Pop_param)

# Important matrices
Important_matrices = (T_stochastic = T_stochastic(P,tau_trans_probs),
                      MC_mat = MC_mat(P))
P = merge(P,Important_matrices)

# Simulation parameters
value_function_iteration_param = (tol = 10^(-16),
                                  max_iter = 10^6)
P = merge(P,value_function_iteration_param)

# Include observed values
Observed_objects = (p_observed = p_observed,
                    r_observed = true_rent,
                    a_observed = Matrix{Float64}(a))
P = merge(P,Observed_objects)

# Add kappa to P and delta_j_tourists to P
κ = DataFrame(CSV.File(main_path*"/data/final/estimates/kappa.csv",header=false))[:,:Column1];
δ_j = DataFrame(CSV.File(main_path*"/data/final/estimates/delta_j.csv",header=false))[:,:Column1];

# Add this to P
P_kappa = (kappa_j = κ, δ_j_tourist = δ_j)
P = merge(P,P_kappa)

# Set path to save output
cd(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file)

################################################################################
# Read objects
################################################################################

##### Airbnb entry CF

## No Airbnb 
r_endo_no_airbnb = DataFrame(CSV.File("equilibrium_objects/r_endo_no_airbnb.csv",header=false))[:,:Column1]
a_endo_no_airbnb = Matrix{Float64}(DataFrame(CSV.File("equilibrium_objects/a_endo_no_airbnb.csv",header=false)))

### Aibnb entry but only prices adjusting
r_exo_a_endo_no_airbnb = DataFrame(CSV.File("equilibrium_objects/r_exo_a_endo_no_airbnb.csv",header = 0))[:,:Column1]
p_exo_a_endo_no_airbnb = DataFrame(CSV.File("equilibrium_objects/p_exo_a_endo_no_airbnb.csv",header = 0))[:,:Column1]

### Airbnb entry, everything adjusting 
r_endo = DataFrame(CSV.File("equilibrium_objects/r_endo.csv",header=false))[:,:Column1]
p_endo = DataFrame(CSV.File("equilibrium_objects/p_endo.csv",header=false))[:,:Column1]
a_endo = Matrix{Float64}(DataFrame(CSV.File("equilibrium_objects/a_endo.csv",header=false)))

####### Homogeneous- heterogenous CF
# Heterogenous, exogenous 
r_exo = DataFrame(CSV.File("equilibrium_objects/r_exo.csv",header=false))[:,:Column1]
p_exo = DataFrame(CSV.File("equilibrium_objects/p_exo.csv",header=false))[:,:Column1]

# Read price sensitivity
δ_r  = vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "log_rent_meter",2:end])))

################################################################################
# Comparison heterogeneous-homogenous case
################################################################################

# Compte marginal utility of consumption
muc = - (1/(1-P.beta)).*(δ_r./(P.income[1:P.K].*P.exp_shares[1:P.K]))

####### Heterogenous case 
## Exogenous amenities -- fixed at the observed level
CS_exo = zeros(P.K,1)
CS_exo[:,1] = welfare_households(r_exo,P.a_observed,P) ./ muc

## Endogenous amenities
CS_endo = zeros(P.K,1)
CS_endo[:,1] = welfare_households(r_endo,a_endo,P) ./ muc

# Save the results
CSV.write("counterfactuals/CS_exo.csv",  DataFrame(CS_exo, :auto), writeheader=false)
CSV.write("counterfactuals/CS_endo.csv",  DataFrame(CS_endo, :auto), writeheader=false)

#################################################################################
## New CE 
#################################################################################

# No airbnb to Airbnb only adjusting prices
CE_1_renter = CE_renter(r_endo_no_airbnb,a_endo_no_airbnb,r_exo_a_endo_no_airbnb,a_endo_no_airbnb,P,δ_r)
CE_1_renter_income = CE_1_renter ./ P.income[1:P.K] * 100
CE_1_homeowner = CE_homeowner(r_endo_no_airbnb,a_endo_no_airbnb,r_exo_a_endo_no_airbnb,a_endo_no_airbnb,P,δ_r)
CE_1_homeowner_income = CE_1_homeowner./ P.income[1:P.K] * 100

# No airbnb to fully endogenous Airbnb 
CE_2_renter = CE_renter(r_endo_no_airbnb,a_endo_no_airbnb,r_endo,a_endo,P,δ_r)
CE_2_renter_income = CE_2_renter ./ P.income[1:P.K] * 100
CE_2_homeowner = CE_homeowner(r_endo_no_airbnb,a_endo_no_airbnb,r_endo,a_endo,P,δ_r)
CE_2_homeowner_income = CE_2_homeowner./ P.income[1:P.K] * 100

CE_renter_hetero = CE_renter(r_exo,P.a_observed,r_endo,a_endo,P,δ_r)

# Store 
df_CE_w_new = DataFrame([CE_1_renter CE_2_renter],["s1","s3"])
CSV.write("counterfactuals/CE_renter_euro.csv", df_CE_w_new)
df_CE_w_new = DataFrame([CE_1_renter_income CE_2_renter_income],["s1","s3"])
CSV.write("counterfactuals/CE_renter_pp.csv", df_CE_w_new)
df_CE_w_new = DataFrame([CE_1_homeowner CE_2_homeowner],["s1","s3"])
CSV.write("counterfactuals/CE_homeowner_euro.csv", df_CE_w_new)
df_CE_w_new = DataFrame([CE_1_homeowner_income CE_2_homeowner_income],["s1","s3"])
CSV.write("counterfactuals/CE_homeowner_pp.csv", df_CE_w_new)

println("########################### End of CF computation ##############################")
println("################################################################################")
