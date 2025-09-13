####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################

println("################################################################################")
println("######################### Perturbation Arbnb solver ############################")
println("####################### Gamma = "*B_file*gamma_files*" ########################################")

println()

# Set path
path = @__DIR__
cd(path)
cd("../../..")
main_path = pwd()

println(pwd())

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
     endogenous_tourist_choices = true,
     airbnb_tax_rate = 0) 

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

# Needed for 
if gamma_files == "081"
    options.λ = 0.95 
end


################################################################################
# Read objects
################################################################################
r_endo = DataFrame(CSV.File("equilibrium_objects/r_endo.csv",header=false))[:,:Column1]
p_endo = DataFrame(CSV.File("equilibrium_objects/p_endo.csv",header=false))[:,:Column1]
a_endo = Matrix{Float64}(DataFrame(CSV.File("equilibrium_objects/a_endo.csv",header=false)))

################################################################################
# Compute equilibrium in the rental market with endogenous amenities
################################################################################

println("##### Solving for endogenous amenities ")

N = 10

# Algorithm options
options.endogenous_r = true
options.endogenous_p = true
options.endogenous_a = true

a_amenity_sol_01 = zeros((N)*P.S,P.J)
r_amenity_sol_01 = zeros(N,P.J)
p_amenity_sol_01 = zeros(N,P.J)

a_amenity_initial_01 = zeros((N)*P.S,P.J)
r_amenity_initial_01 = zeros(N,P.J)
p_amenity_initial_01 = zeros(N,P.J)

for i in 1:N
    lb = (i-1)*6 + 1
    ub = (i-1)*6 + 6

    initial_r = P.r_observed + rand(Uniform(-0.1,0.1),P.J).*P.r_observed 
    initial_p = P.p_observed + rand(Uniform(-0.1,0.1),P.J).*P.p_observed 
    initial_a = P.a_observed + rand(Uniform(-0.1,0.1),P.J,P.S).*P.a_observed 
    
    a_amenity_initial_01[lb:ub,:] = initial_a'
    p_amenity_initial_01[i,:] = initial_p
    r_amenity_initial_01[i,:] = initial_r


    @time r_eq_full, p_eq_full, a_eq_full, EV, DL = full_solver(initial_r,initial_p,initial_a,P,options)


    a_amenity_sol_01[lb:ub,:] = a_eq_full'
    p_amenity_sol_01[i,:] = p_eq_full
    r_amenity_sol_01[i,:] = r_eq_full

    dist_a = norm(a_endo-a_eq_full,Inf)
    dist_r = norm(r_endo-r_eq_full,Inf)
    dist_p = norm(p_endo-p_eq_full,Inf)


    println("Sample ", i," completed with distances: a = ",dist_a,", p = ",dist_p,", r = ", dist_r)
    println()

    # Store
    CSV.write("counterfactuals/r_sol_stability_01.csv",  Tables.table(r_amenity_sol_01), writeheader=false)
    CSV.write("counterfactuals/p_sol_stability_01.csv",  Tables.table(p_amenity_sol_01), writeheader=false)
    CSV.write("counterfactuals/a_sol_stability_01.csv",  Tables.table(a_amenity_sol_01), writeheader=false)
    CSV.write("counterfactuals/r_initial_stability_01.csv",  Tables.table(r_amenity_initial_01), writeheader=false)
    CSV.write("counterfactuals/p_initial_stability_01.csv",  Tables.table(p_amenity_initial_01), writeheader=false)
    CSV.write("counterfactuals/a_initial_stability_01.csv",  Tables.table(a_amenity_initial_01), writeheader=false)

end


r_amenity_sol_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/r_sol_stability_01.csv",header=false)))
p_amenity_sol_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/p_sol_stability_01.csv",header=false)))
a_amenity_sol_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/a_sol_stability_01.csv",header=false)))

r_amenity_initial_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/r_initial_stability_01.csv",header=false)))
p_amenity_initial_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/p_initial_stability_01.csv",header=false)))
a_amenity_initial_01 = Matrix{Float64}(DataFrame(CSV.File("counterfactuals/a_initial_stability_01.csv",header=false)))

mean_diff_ini = zeros(N,9)
mean_diff_sol = zeros(N,9)

mean_diff_abs_ini = zeros(N,9)
mean_diff_abs_sol = zeros(N,9)

norminf_diff_ini = zeros(N,4)
norminf_diff_sol = zeros(N,4)

for i = 1:N
    lb = (i-1)*6 + 1
    ub = (i-1)*6 + 6

    mean_diff_abs_sol[i,:] = [i mean(abs.(r_amenity_sol_01[i,:]-r_endo)./r_endo) mean(abs.(p_amenity_sol_01[i,:]-p_endo)./p_endo)    mean(abs.(a_amenity_sol_01[lb:ub,:]'-a_endo)./a_endo, dims = 1)]
    mean_diff_abs_ini[i,:] = [i mean(abs.(r_amenity_initial_01[i,:]-P.r_observed )./P.r_observed ) mean(abs.(p_amenity_initial_01[i,:]-P.p_observed )./P.p_observed )    mean(abs.(a_amenity_initial_01[lb:ub,:]'-P.a_observed )./P.a_observed, dims = 1)]
    
    mean_diff_sol[i,:] = [i mean((r_amenity_sol_01[i,:]-r_endo)./r_endo) mean((p_amenity_sol_01[i,:]-p_endo)./p_endo)    mean((a_amenity_sol_01[lb:ub,:]'-a_endo)./a_endo, dims = 1)]
    mean_diff_ini[i,:] = [i mean((r_amenity_initial_01[i,:]-P.r_observed )./P.r_observed ) mean((p_amenity_initial_01[i,:]-P.p_observed )./P.p_observed )    mean((a_amenity_initial_01[lb:ub,:]'-P.a_observed )./P.a_observed, dims = 1)]

    norminf_diff_sol[i,:] = [i norm(abs.(r_amenity_sol_01[i,:]-r_endo)./r_endo, Inf) norm(abs.(p_amenity_sol_01[i,:]-p_endo)./p_endo, Inf)    norm(abs.(a_amenity_sol_01[lb:ub,:]'-a_endo)./a_endo, Inf)]
    norminf_diff_ini[i,:] = [i norm(abs.(r_amenity_initial_01[i,:]-P.r_observed )./P.r_observed, Inf ) norm(abs.(p_amenity_initial_01[i,:]-P.p_observed )./P.p_observed,  Inf )    norm(abs.(a_amenity_initial_01[lb:ub,:]'-P.a_observed )./P.a_observed,  Inf)]
end

CSV.write("counterfactuals/mean_diff_ini_stability.csv",  Tables.table(mean_diff_ini), writeheader=false)
CSV.write("counterfactuals/mean_diff_sol_stability.csv",  Tables.table(mean_diff_sol), writeheader=false)

CSV.write("counterfactuals/norminf_diff_ini_stability.csv",  Tables.table(norminf_diff_ini), writeheader=false)
CSV.write("counterfactuals/norminf_diff_sol_stability.csv",  Tables.table(norminf_diff_sol), writeheader=false)
