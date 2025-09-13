####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################

println("################################################################################")
println("######################### Initializing Airbnb tax CF ###########################")
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

options.λ = 0.85

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
              beta = 0.85)

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
                amenity_resid = amenity_supply_residuals,
                norm_amenities = norm_amenities)
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
pwd()
################################################################################
# Read objects 
################################################################################

r_endo = DataFrame(CSV.File("equilibrium_objects/r_endo.csv",header=false))[:,:Column1]
p_endo = DataFrame(CSV.File("equilibrium_objects/p_endo.csv",header=false))[:,:Column1]
a_endo = Matrix{Float64}(DataFrame(CSV.File("equilibrium_objects/a_endo.csv",header=false)))

# Compte marginal utility of consumption
δ_r  = vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "log_rent_meter",2:end])))
muc = - (1/(1-P.beta)).*(δ_r./(P.income[1:P.K].*P.exp_shares[1:P.K]))
    
################################################################################
# Loop over Airbnb tax
################################################################################

# Algorithm options
options.endogenous_r = true
options.endogenous_p = true
options.endogenous_a = true
options.airbnb_tax_rate = 0.0
options.amenity_tax_rate = 0.0

N = 9

tax_grid = [0.01*i for i in 0:N]

println("##### Solving for airbnb tax rate ")
println()

a_sol = zeros((N+1)*P.S,P.J)
r_sol = zeros(N+1,P.J)
p_sol = zeros(N+1,P.J)

lb = 1
ub = 6
a_sol[lb:ub,:] = copy(a_endo')
p_sol[1,:] = copy(p_endo)
r_sol[1,:] = copy(r_endo)

CS_airbnb_tax = zeros(N,1+P.K)
CS_airbnb_tax_pp = zeros(N,1+P.K)


for i in 1:2

    lb = (i-1)*6 + 1
    ub = (i-1)*6 + 6

    options.airbnb_tax_rate = tax_grid[i]
    initial_r = r_sol[i,:]
    initial_p = p_sol[i,:]
    initial_a = Matrix{Float64}(a_sol[lb:ub,:]')
    @time r_eq_full_airbnb, p_eq_full_airbnb, a_eq_full_airbnb, EV, DL = full_solver(initial_r,initial_p,initial_a,P,options)

    a_sol[i*6 + 1:i*6 + 6,:] = a_eq_full_airbnb'
    p_sol[i+1,:] = p_eq_full_airbnb
    r_sol[i+1,:] = r_eq_full_airbnb

    CS_airbnb_tax[i,2:end] =  welfare_households(r_eq_full_airbnb,a_eq_full_airbnb,P) ./ muc
    CS_airbnb_tax_pp[i,2:end] =  100*(welfare_households(r_eq_full_airbnb,a_eq_full_airbnb,P) ./ muc)./(P.income[1:P.K])

    println("Tax rate ", tax_grid[i]," completed!")
    println()

end

CS_airbnb_tax[:,1] = tax_grid[1:end-1]
CS_airbnb_tax_pp[:,1] = tax_grid[1:end-1]

# Store
CSV.write("counterfactuals/r_sol_airbnb_tax.csv",  Tables.table(r_sol), writeheader=false)
CSV.write("counterfactuals/p_sol_airbnb_tax.csv",  Tables.table(p_sol), writeheader=false)
CSV.write("counterfactuals/a_sol_airbnb_tax.csv",  Tables.table(a_sol), writeheader=false)
CSV.write("counterfactuals/CS_airbnb_tax.csv",  Tables.table(CS_airbnb_tax), writeheader=false)
CSV.write("counterfactuals/CS_pp_airbnb_tax.csv",  Tables.table(CS_airbnb_tax_pp), writeheader=false)


################################################################################
# Loop over Touristic Amenity tax
################################################################################

# Algorithm options
options.endogenous_r = true
options.endogenous_p = true
options.endogenous_a = true
options.airbnb_tax_rate = 0.0
options.amenity_tax_rate = 0.0

N = 9

tax_grid = [0.01*i for i in 0:N]

println("##### Solving for amenity tax rate ")
println()

a_amenity_sol = zeros((N+1)*P.S,P.J)
r_amenity_sol = zeros(N+1,P.J)
p_amenity_sol = zeros(N+1,P.J)

a_amenity_sol[1:P.S,:] = copy(a_endo')
p_amenity_sol[1,:] = copy(p_endo)
r_amenity_sol[1,:] = copy(r_endo)

CS_amenity_tax = zeros(N,1+P.K)
CS_amenity_pp = zeros(N,1+P.K)


for i in 1:2
    lb = (i-1)*6 + 1
    ub = (i-1)*6 + 6

    options.amenity_tax_rate = tax_grid[i]
    initial_r = copy(r_amenity_sol[i,:])
    initial_p = copy(p_amenity_sol[i,:])
    initial_a = copy(Matrix{Float64}(a_amenity_sol[lb:ub,:]'))
    @time r_eq_full_amenity, p_eq_full_amenity, a_eq_full_amenity EV, DL = full_solver(initial_r,initial_p,initial_a,P,options)


    a_amenity_sol[i*6 + 1:i*6 + 6,:] = a_eq_full_amenity'
    p_amenity_sol[i+1,:] = p_eq_full_amenity
    r_amenity_sol[i+1,:] = r_eq_full_amenity

    CS_amenity_tax[i,1] = tax_grid[i]
    CS_amenity_tax[i,2:end] =  welfare_households(r_eq_full_amenity,a_eq_full_amenity,P) ./ muc
    CS_amenity_pp[i,2:end] =  100*(welfare_households(r_eq_full_amenity,a_eq_full_amenity,P) ./ muc)./(P.income[1:P.K])

    println("Tax rate ", tax_grid[i]," completed!")
    println()

end

CS_amenity_pp[:,1] = tax_grid[1:end-1]

# Store
CSV.write("counterfactuals/r_sol_amenity_tax.csv",  Tables.table(r_amenity_sol), writeheader=false)
CSV.write("counterfactuals/p_sol_amenitytax.csv",  Tables.table(p_amenity_sol), writeheader=false)
CSV.write("counterfactuals/a_sol_amenity_tax.csv",  Tables.table(a_amenity_sol), writeheader=false)
CSV.write("counterfactuals/CS_amenity_tax.csv",  Tables.table(CS_amenity_tax), writeheader=false)
CSV.write("counterfactuals/CS_pp_amenity_tax.csv",  Tables.table(CS_amenity_pp), writeheader=false)


