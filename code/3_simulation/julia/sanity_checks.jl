####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

println("################################################################################")
println("########################### Starting sanity checks #############################")
println()

################################################################################
# Set-up
################################################################################

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
     endogenous_tourist_choices = true,
     airbnb_tax_rate = 0) 

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


# Include observed values
Observed_objects = (p_observed = p_observed,
                    r_observed = true_rent,
                    a_observed = Matrix{Float64}(a))
P = merge(P,Observed_objects)

# Add kappa to P and delta_j_tourists to P
κ = DataFrame(CSV.File(main_path*"/data/final/estimates/kappa.csv",header=false))[:,:Column1];
δ_j = DataFrame(CSV.File(main_path*"/data/final/estimates/delta_j.csv",header=false))[:,:Column1];

# Add this to P
P_kappa = (kappa_j = κ, δ_j_tourist = δ_j);
P = merge(P,P_kappa);

################################################################################
# Sanity checks
################################################################################

### Supply 
r = P.r_observed
p = P.p_observed
ltr_hat = S_L(r,p,P,options)
ltr_share = housing_supply_data.quantity_ltr./(housing_supply_data.quantity_ltr+ housing_supply_data.quantity_str)

println("Difference in predicted and observed housing supply: ", norm(ltr_share - ltr_hat,Inf))


### Demand tourists 
tourist_guest_demand_hat = tourist_guest_demand(p,a,P)
tourist_guest_demand_observed = tourist_demand_est_df.pop_tourist
println("Difference in predicted and observed tourist demand: ", norm(tourist_guest_demand_hat - tourist_guest_demand_observed,Inf))

DS_hat = D_S(r,p,a,P)
str_total_guests = panel_covariates_demand.pop_tourists_airbnb
println("Difference in predicted and observed tourist guests: ", norm(DS_hat - str_total_guests,Inf))


#### Amenities
# Compute predicted amenities at the observed demand
r_observed = P.r_observed
p_observed = P.p_observed
D = type_counts_inner[:,1:3]
a_observed = P.a_observed
a_j = vec(sum(a_observed ,dims=2))
pred_amenities_observed = Amenity_supply(r_observed,p_observed,a_observed,D,P)

# Check
println("Difference in predicted and observed amenities: ", norm(pred_amenities_observed - a_observed,Inf))

#### Demand for endogenous types
# Read in u_hat
cd(main_path*"/data/final/")
u_hat_df = DataFrame(CSV.File("estimates/demand_location_choice_u_hat"*myopic_file*static_file*".csv"))
u_hat_df = u_hat_df[u_hat_df.year .== 2017,:]
u_hat_df = u_hat_df[(u_hat_df.gb .>= 1) .&  (u_hat_df.gb .<= P.J),:]
u_hat_df = u_hat_df[(u_hat_df.combined_cluster .>= 1) .&  (u_hat_df.combined_cluster .<= P.K),:]
sort!(u_hat_df,[:combined_cluster,:gb])
u_hat_vec = vec(u_hat_df[!,"u_hat"])
u_hat = reshape(u_hat_vec,P.J,P.K)

u_hat_mat = utility(r_observed,P.a_observed,P)

for k in 1:P.K
    println("Difference in predicted and observed u_hat for type ", k ,": ", norm(u_hat_mat[1:P.J,k] - u_hat[:,k],Inf))
end

demand_sq_area = D_sq_area(r_observed,P)
EVs_initial = ones(P.tau_bar*(P.J+1),P.K)
DL_mat,EVs_g = D_L(r_observed,a_observed,P,EVs_initial)

DL_mat = DL_mat[1:P.J,:]
ED_L_norm(r_observed,p_observed,a_observed,P);
println()
