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
include("model_endogenous_tourists.jl")

# Seed
Random.seed!(1234)

#### Basic dimensions
P = (K = 3,
     KT = 7,
     S = 6,
     J = 22,
     tau_bar = 2)

cd(main_path*"/data/final/")
#println(pwd())

# Algorithm options
mutable struct algo_options 
    EVs_initial::Matrix{Float64}
    δ_prices::Float64
    δ_amenities::Float64
    λ::Float64
    tol_r::Float64
    tol_p::Float64
    tol_a::Float64
    tol_EV::Float64
    max_iter_r::Float64
    max_iter_p::Float64
    max_iter_a::Float64
    max_iter_EV::Float64
    endogenous_r::Bool
    endogenous_p::Bool
    endogenous_a::Bool
    airbnb_tax_rate::Float64
    amenity_tax_rate::Float64
end

options = algo_options(
                        ones(P.tau_bar*(P.J+1),P.K),
                        0.01,
                        0.2,
                        0.85,
                        10^(-12),
                        10^(-12),
                        10^(-10),
                        10^(-20),
                        5000,
                        5000,
                        2000,
                        10^6,
                        false,
                        false,
                        false,
                        0,
                        0)

################################################################################
# Read in estimated coefficients
################################################################################

#### DEMAND ESTIMATES
# Residential choice
estimates_demand = CSV.read("estimates/ivreg_demand_location_choice_estimates"*myopic_file*static_file*".csv",DataFrame)
estimates_demand = estimates_demand[2:end,:]
list_demand = vec(estimates_demand[:,1])
θ_resident = Matrix{Float64}(parse.(Float64, estimates_demand[1:end,2:end]))
amenities = ["log_amenity_1", "log_amenity_2", "log_amenity_3", "log_amenity_4", "log_amenity_5","log_amenity_6"]
gamma_0_coef = ifelse(static,zeros(P.K), vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "gamma_0_vec",2:end]))))
gamma_1_coef = ifelse(static,zeros(P.K),vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "gamma_1_vec",2:end]))))
gamma_2_coef = ifelse(static,zeros(P.K),vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "gamma_2_vec",2:end]))))
tau_coef = ifelse(static,zeros(P.K),vec(parse.(Float64,Matrix(estimates_demand[estimates_demand.Column1 .== "tau",2:end]))))


#### HOUSING SUPPLY ESTIMATES
# Read supply estimates
estimates_supply = CSV.read("estimates/housing_supply_estimates.csv",DataFrame)
alpha_housing_supply = mean(estimates_supply.alpha)


#### AMENITY ESTIMATES
# Amenity supply estimates
gamma_val = parse(Float64, gamma_files)/100
estimates_amenity_supply = CSV.read("estimates/"*B_file*"amenity_supply_estimation_gamma_"*@sprintf("%.2f", gamma_val)*".csv",DataFrame)
alpha_amenities = estimates_amenity_supply[1:P.S*P.KT,:x1]
alpha_estimates = reshape(alpha_amenities,P.KT,P.S)
time_FE_y_amenity_supply = estimates_amenity_supply[end,:x1]
location_FE_amenity_supply = [0;estimates_amenity_supply[(P.S*P.KT)+1:P.S*P.KT+P.J-1,:x1]]
df_residuals_amenity_supply = CSV.read("estimates/"*B_file*"amenity_supply_residuals_gamma_"*@sprintf("%.2f", gamma_val) *".csv",DataFrame)
amenity_supply_residuals = df_residuals_amenity_supply[df_residuals_amenity_supply.x3.== year,:x4]
amenity_supply_residuals = reshape(amenity_supply_residuals,P.J,P.S)
norm_amenities = mean(Vector{Float64}(df_residuals_amenity_supply.x5))

#### TOURIST ESTIMATES
# Read in demand estimates
estimates_tourists = CSV.read("estimates/tourist_demand_estimates.csv",DataFrame)
list_estimates_tourists = vec(estimates_tourists[:,1])
θ_tourist = vec(estimates_tourists[:,2])
tourist_demand_price_coef = θ_tourist[1]
tourist_demand_amenity_coef = θ_tourist[2:end-1]
tourist_demand_controls = θ_tourist[end]

################################################################################
# Read data in
################################################################################

#### HOUSING SUPPLY
# Supply covariates 
housing_supply_data = DataFrame(CSV.File("inputs/str_ltr_gebied.csv"))
sort!(housing_supply_data,:gb_code)

##### NEIGHBORHOOD CHARACTERISTICS
# Demand covariates
panel_covariates_demand = CSV.read("inputs/gebied_covariates_panel.csv",DataFrame)
panel_covariates_demand = panel_covariates_demand[panel_covariates_demand.year .== year,:]
sort!(panel_covariates_demand,:gb)

# Select only necesary variables
X_resident = panel_covariates_demand
X_resident[:, :tau] .= 0
X_resident[:, :gamma_0_vec] .= 0
X_resident[:, :gamma_1_vec] .= 0
X_resident[:, :gamma_2_vec] .= 0
X_resident = X_resident[:,list_demand]
X_full = Matrix{Float64}(X_resident)
X_resident_exo = X_resident[:,setdiff(list_demand,vcat(amenities, "log_rent_meter"))]
X_resident_exo = Matrix{Float64}(X_resident_exo)

##### EXPENDITURE SHARES
exp_shares_df = CSV.read("inputs/expenditure_shares.csv",DataFrame)
exp_shares = exp_shares_df.exp_sh_c

#### ANNUAL INCOME
annual_income_df = CSV.read("inputs/annual_income.csv",DataFrame)
annual_income_df = annual_income_df[annual_income_df.year .== year,:]
sort!(annual_income_df,:combined_cluster)
yearly_disp_income = annual_income_df[!,"disposable_income"]

##### POPULATION COUNTS
# Population counts at the gebied level
type_counts_gb_df = DataFrame(CSV.File("inputs/gebied_population_counts_panel.csv"))
filter!(row->!ismissing(row.combined_cluster), type_counts_gb_df)
filter!(row->!ismissing(row.year), type_counts_gb_df)
filter!(row->!ismissing(row.gb), type_counts_gb_df)
type_counts_gb_df = type_counts_gb_df[type_counts_gb_df.year .== year,:]
sort!(type_counts_gb_df,[:combined_cluster, :gb])
type_gb_counts = reshape(type_counts_gb_df[!,"num_hh"],P.J+1,P.KT-1)
type_counts_inner = Matrix{Float64}(type_gb_counts[1:end-1,:])
exo_type_counts = type_counts_inner[:,4:6]

# Total population counts 
type_counts = sum(type_gb_counts,dims = 1)
type_counts = vec(type_counts[1:P.K])


#### TRANSITIOM PROBABILITIES
tau_trans_probs = DataFrame(CSV.File("inputs/tau_transition_probs_all.csv"))
tau_trans_probs = tau_trans_probs[tau_trans_probs.year .== year,:]
replace!(tau_trans_probs.gb, 0 => P.J+1)
sort!(tau_trans_probs,[:combined_cluster,:gb])

#### TOURISTS
# Tourist covariates 
tourist_demand_est_df = DataFrame(CSV.File("inputs/gebied_tourist_demand_covariates.csv"))
tourist_demand_controls_data = tourist_demand_est_df.log_review_scores_location
mean_accommodates = tourist_demand_est_df.mean_accommodates[1:P.J] 

# Tourist population
gebied_tourist_population = vec(panel_covariates_demand[!,"pop_tourists_total"])
gebied_tourist_population_hotels = vec(panel_covariates_demand[!,"pop_tourists_hotels"])

#### DISTANCE MATRIX
dist_mat = Matrix{Float64}(DataFrame(CSV.File("inputs/dist_mat_centroids.csv")))

##### CONSTRUCT VARIABLES
# Housing stock
houses_LT = panel_covariates_demand.tenancy_status_1
houses_landlords = housing_supply_data.quantity_ltr +  housing_supply_data.quantity_str
# squared footage
avg_squared_footage = exp.(panel_covariates_demand.log_area_by_usage_1)
# prices
true_rent = panel_covariates_demand.unit_rent./avg_squared_footage
p_observed = housing_supply_data.price_str/365
# amenities
a = exp.(Matrix{Float64}(panel_covariates_demand[:,amenities]))
# Airbnb listings and conversion factor to guests
n_listings = housing_supply_data.quantity_str
listings_to_total_str_guests = panel_covariates_demand.pop_tourists_airbnb./n_listings
str_guests_to_total_guests = panel_covariates_demand.pop_tourists_airbnb[1:P.J]./tourist_demand_est_df.pop_airbnb_commercial[1:P.J] ## later on we need to correct this
hotel_beds = panel_covariates_demand.hotel_beds
total_tourists = tourist_demand_est_df.total_tourist_pop[1]

################################################################################
# Unpack and define parameters
################################################################################

# Basic Parameters
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
Important_matrices = (T_stochastic_new = T_stochastic(P,tau_trans_probs),
                      MC_mat = MC_mat(P))
P = merge(P,Important_matrices)

# Simulation parameters
value_function_iteration_param = (tol = 10^(-16),
                                  max_iter = 10^6)
P = merge(P,value_function_iteration_param)

################################################################################
# Include observed values
################################################################################
Observed_objects = (p_observed = p_observed,
                    r_observed = true_rent,
                    a_observed = Matrix{Float64}(a))

P = merge(P,Observed_objects)

################################################################################
# Calibrate kappa and delta_j
################################################################################

# kappa for housing supply
ltr_share = housing_supply_data.quantity_ltr./(housing_supply_data.quantity_ltr+ housing_supply_data.quantity_str)
srt_share = 1 .- ltr_share
price_gap = P.r_observed.*P.avg_squared_footage - P.p_observed*365
κ = log.(ltr_share)- log.(srt_share)-P.alpha*price_gap./10^4

# delta for tourist demand
prob_tourists = tourist_demand_est_df.prob_tourists
p_guest = P.p_observed./P.mean_accommodates
y_hat_tourists = [[log.(p_guest);0] [log.(a); zeros(6)'] P.tourist_demand_controls]*P.θ_tourist
δ_j = log.(prob_tourists) .- log(prob_tourists[end]) - y_hat_tourists

# Add this to P
P_kappa = (kappa_j = κ, δ_j_tourist = δ_j)
P = merge(P,P_kappa)

CSV.write("estimates/kappa.csv",  Tables.table(κ), writeheader=false)
CSV.write("estimates/delta_j.csv",  Tables.table(δ_j), writeheader=false)


################################################################################
# Create folders to save output
################################################################################

mkpath(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file)
mkpath(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file*"/temp")
mkpath(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file*"/figures")
mkpath(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file*"/equilibrium_objects")
mkpath(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file*"/counterfactuals")


# Set path to save output
cd(main_path*"/data/simulation_results/gamma_"*B_file*gamma_files*myopic_file*static_file)