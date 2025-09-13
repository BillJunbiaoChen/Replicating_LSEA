####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################


println("################################################################################")
println("###################### Initializing Homogeneous solver #########################")
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

# Seed
Random.seed!(1234)

options.λ = 0.95

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
Pop = type_counts[1:3]
mean_θ_resident = mean(θ_resident, dims = 2,weights(Pop)).*ones(size(θ_resident,1),P.K)
homogeneous_θ_resident = θ_resident
homogeneous_θ_resident[2:7,:] = mean_θ_resident[2:7,:]
Util_param = (θ_resident = θ_resident,
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
# Solve for both housing prices with exogenous amenities
################################################################################

# Initializing options
options.endogenous_r = true
options.endogenous_p = true
options.endogenous_a = false

initial_r = P.r_observed
initial_p = P.p_observed
initial_a = P.a_observed

# Solver
println("##### Solving for exogenous amenities ")
@time r_eq_exo, p_eq_exo, a_eq_fexo, EV_exo, DL_exo = full_solver(initial_r,initial_p,initial_a,P,options)

df = (r_eq_exo = r_eq_exo, initial_r  = initial_r)
model = lm(@formula(r_eq_exo ~ 1 + initial_r), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_r,r_eq_exo, legend = false, 
    xlabel = "Observed rent per square meter (annual)", ylabel = "Eq. rent per square meter (annual)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_r, predict(model, df))
#savefig("figures/r_exo_homogenous")
println()
println("Algorithm finished!")
println("Rent fit values: r2 = ",r2m,", beta = ",coefs_m[2])

# Compare to observed airbnb prices
df = (p_eq_exo = p_eq_exo, initial_p  = initial_p)
model = lm(@formula(p_eq_exo ~ 1 + initial_p), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_p,p_eq_exo, legend = false, 
    xlabel = "Observed Airbnb daily prices", ylabel = "Eq. Airbnb daily prices",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_p, predict(model, df))
#savefig("figures/p_exo_homogeneous")
println("Airbnb prices fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)
println()

# Store
CSV.write("equilibrium_objects/r_exo_homogeneous.csv",  Tables.table(r_eq_exo), writeheader=false)
CSV.write("equilibrium_objects/p_exo_homogeneous.csv",  Tables.table(p_eq_exo), writeheader=false)
CSV.write("equilibrium_objects/DL_exo_homogeneous.csv",  Tables.table(DL_exo[1:P.J,:]), writeheader=false)


################################################################################
# Compute equilibrium in the rental market with endogenous amenities
################################################################################

# Initializing options
options.endogenous_r = true
options.endogenous_p = true
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
    xlabel = "Observed rent per square meter (annual)", ylabel = "Eq. rent per square meter (annual)",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_r, predict(model, df))
#savefig("figures/r_endo_homogeneous")
println()
println("Algorithm finished!")
println("Rent fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)

# Compare to observed airbnb prices
df = (p_eq_full = p_eq_full, initial_p  = initial_p)
model = lm(@formula(p_eq_full ~ 1 + initial_p), df)
r2m = r2(model)
coefs_m = coef(model)
rounded_r2 = round(r2m,digits=3)
intercept_rounded = round(coefs_m[1],digits=3)
slope_rounded = round(coefs_m[2],digits=3)
scatter(initial_p,p_eq_full, legend = false, 
    xlabel = "Observed Airbnb daily prices", ylabel = "Eq. Airbnb daily prices",
    title = "Slope = $(slope_rounded), Intercept = $(intercept_rounded) \\n R2 = $(rounded_r2)")
plot!(initial_p, predict(model, df))
#savefig("figures/p_endo_homogeneous")
println("Airbnb pirces fit values: r2 = ",rounded_r2,", beta = ",slope_rounded)


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
    #savefig("figures/a_endo_homogeneous_s$(s)")
end


# Additional checks to make sure this is an equilibrium
DL,EVs_g = D_L(r_eq_full,a_eq_full,P)
DL = DL[1:P.J,:]
(norm(ED_L_norm(r_eq_full,p_eq_full,a_eq_full,P),Inf))

# Store
CSV.write("equilibrium_objects/r_endo_homogeneous.csv",  Tables.table(r_eq_full), writeheader=false)
CSV.write("equilibrium_objects/p_endo_homogeneous.csv",  Tables.table(p_eq_full), writeheader=false)
CSV.write("equilibrium_objects/a_endo_homogeneous.csv",  Tables.table(a_eq_full), writeheader=false)
CSV.write("equilibrium_objects/DL_endo_homogeneous.csv",  Tables.table(DL), writeheader=false)


################################################################################
# Computer Welfare for homogenenous-heterogenius comparison
################################################################################

# Read objects  
r_exo_homogeneous = DataFrame(CSV.File("equilibrium_objects/r_exo_homogeneous.csv",header=false))[:,:Column1]
p_exo_homogeneous = DataFrame(CSV.File("equilibrium_objects/p_exo_homogeneous.csv",header=false))[:,:Column1]
r_endo_homogeneous = DataFrame(CSV.File("equilibrium_objects/r_endo_homogeneous.csv",header=false))[:,:Column1]
a_endo_homogeneous = Matrix{Float64}(DataFrame(CSV.File("equilibrium_objects/a_endo_homogeneous.csv",header=false)))
p_endo_homogeneous = DataFrame(CSV.File("equilibrium_objects/p_endo_homogeneous.csv",header=false))[:,:Column1]

# Compute marginal utility of consumption
δ_r  = vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "log_rent_meter",2:end])))
muc = - (1/(1-P.beta)).*(δ_r./(P.income[1:P.K].*P.exp_shares[1:P.K]))

## Exogenous amenities -- fixed at the observed level
CS_exo_homogeneous = zeros(P.K,1)
CS_exo_homogeneous[:,1] = welfare_households(r_exo_homogeneous,P.a_observed,P) ./ muc

## Endogenous amenities
CS_endo_homogeneous = zeros(P.K,1)
CS_endo_homogeneous[:,1] = welfare_households(r_endo_homogeneous,a_endo_homogeneous,P) ./ muc

#### Save the results
CSV.write("counterfactuals/CS_exo_homogeneous.csv",  DataFrame(CS_exo_homogeneous, :auto), writeheader=false)
CSV.write("counterfactuals/CS_endo_homogeneous.csv",  DataFrame(CS_endo_homogeneous, :auto), writeheader=false)

println()
