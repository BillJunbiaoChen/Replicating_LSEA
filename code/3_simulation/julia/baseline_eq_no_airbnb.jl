####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################

println("################################################################################")
println("######################## Initializing No Arbnb solver ##########################")
println("####################### Gamma = "*B_file*gamma_files*" ########################################")

println()

# Set path
path = @__DIR__
cd(path)
cd("../../..")
main_path = pwd()
print(pwd())

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
     include_airbnb = false,
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
# Compute equilibrium in the rental market with exogenous amenities
################################################################################

# Initializing options
options.endogenous_r = true
options.endogenous_p = false
options.endogenous_a = false

initial_r = P.r_observed
initial_p = P.p_observed
initial_a = P.a_observed

# Solver
println()
println("##### Solving for exogenous amenities ")
@time r_eq_exo, p_eq_exo, a_eq_fexo, EV_exo, DL_exo = full_solver(initial_r,initial_p,initial_a,P,options)

# Compare to observed rent
df = (r_eq_exo = r_eq_exo, initial_r  = initial_r)
model = lm(@formula(r_eq_exo ~ 1 + initial_r), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_r,r_eq_exo,legend = false, 
    xlabel = "Observed rent (per area unit & year)", ylabel = "Eq. (per area unit & year)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_r, predict(model, df))
println()
#savefig("figures/r_exo_no_airbnb")
println("Algorithm finished!")
println("Rent fit values: r2 = ",r2m,", beta = ",coefs_m[2])
println()

# Compare to observed rent
df = (p_eq_exo = p_eq_exo, initial_p  = initial_p)
model = lm(@formula(p_eq_exo ~ 1 + initial_p), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_p,p_eq_exo, legend = false, 
    xlabel = "Observed rent (per area unit & year)", ylabel = "Eq. (per area unit & year)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_p, predict(model, df))
println()
println("Airbnb pirces fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)

# Store

CSV.write("equilibrium_objects/r_exo_no_airbnb.csv",  Tables.table(r_eq_exo), writeheader=false)
CSV.write("equilibrium_objects/DL_exo_no_airbnb.csv",  Tables.table(DL_exo), writeheader=false)

################################################################################
# Compute equilibrium in the rental market with endogenous amenities
################################################################################

# Initializing options
options.endogenous_r = true
options.endogenous_p = false
options.endogenous_a = true

initial_r = P.r_observed
initial_p = P.p_observed
initial_a = P.a_observed

println("##### Solving for endogenous amenities ")
@time r_eq_full, p_eq_full, a_eq_full, EV, DL = full_solver(initial_r,initial_p,initial_a,P,options)

# Compare to observed rent
df = (r_eq_full = r_eq_full, initial_r  = initial_r)
model = lm(@formula(r_eq_full ~ 1 + initial_r), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_r,r_eq_full, legend = false, 
    xlabel = "Observed rent (per area unit & year)", ylabel = "Eq. (per area unit & year)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_r, predict(model, df))
#savefig("figures/r_endo_no_airbnb")
println()
println("Algorithm finished!")
println("Rent prices fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)

# Compare to observed rent
df = (p_eq_full = p_eq_full, initial_p  = initial_p)
model = lm(@formula(p_eq_full ~ 1 + initial_p), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_p,p_eq_full, legend = false, 
    xlabel = "Observed rent (per area unit & year)", ylabel = "Eq. (per area unit & year)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_p, predict(model, df))
println("Airbnb prices fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)


# Compare to observed amenities
amenity_names_map = ["tourism_offices","restaurants_locations","bars_locations","food_stores","nonfood_stores","nurseries"]
for s in 1:P.S
    df_a = (a_eq_c = a_eq_full[:,s], initial_a_c  = initial_a[:,s])
    model_a = lm(@formula(a_eq_c ~ 1 + initial_a_c), df_a)
    r2m_a = r2(model_a)
    coefs_m_a = coef(model_a)
    rounded_r2_a = round(r2m_a,digits=3)
    intercept_rounded_a = round(coefs_m_a[1],digits=3)
    slope_rounded_a = round(coefs_m_a[2],digits=3)
    scatter(initial_a[:,s],a_eq_full[:,s], legend = false, 
        xlabel = "Observed $(amenity_names_map[s])", ylabel = "Eq. $(amenity_names_map[s])",
        title = "Slope = $(slope_rounded_a), Intercept = $(intercept_rounded_a) \\n R2 = $(rounded_r2_a)")
    plot!(initial_a[:,s], predict(model_a, df_a))
    println("Amenity ", s," fit values: r2 = ",rounded_r2_a,", beta = ",slope_rounded_a)
    #savefig("figures/a_endo_s$(s)_no_airbnb")
end

# Store
CSV.write("equilibrium_objects/r_endo_no_airbnb.csv",  Tables.table(r_eq_full), writeheader=false)
CSV.write("equilibrium_objects/a_endo_no_airbnb.csv",  Tables.table(a_eq_full), writeheader=false)
CSV.write("equilibrium_objects/DL_endo_no_airbnb.csv",  Tables.table(DL), writeheader=false)

println()