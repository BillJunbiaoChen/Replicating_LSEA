using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Distributed

include("MC_functions_full_dynamic_corrected_stochastic_loc_cap.jl")

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Fix seed for reproducibility
Random.seed!(1234)

# Base directory for data storage
const BASE_DIR = "../Data"

# Fixed parameters
const S = 2
const TT = 10
const tau_bar = 2
const J = 25
const P_mat = Matrix{Float64}([0.7 0.3; 0 1])

# Parameter settings
const populations = [50000, 1000050]
const xi_types = ["endogenous", "exogenous", "zero"]
const fixed_effects = [true]
const tau_dummy = [true]

# True parameters
const α = -0.05
const β = 0.1*ones(Float64,S)
const gamma_0 = -0.0025
const gamma_1 = -0.0025
const gamma_2 = -0.5
const δ = 0.1
const true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]

# First set up the location and time dummies as globals
# Compute the matrix of location dummies for the utility function
M_J = vcat(Matrix{Int64}(I, J-2,J-2),zeros(Int64,2,J-2))
Dm_J = repeat(M_J,inner=(TT,1))
global loc_dummy_mat = copy(Dm_J)

# Compute the matrix of time dummies for the utility function
M_T = vcat(Matrix{Int64}(I, TT-1,TT-1),zeros(Int64,1,TT-1))
Dm_T = vcat(repeat(M_T,outer = (J-1,1)),zeros(TT,TT-1))
global time_dummy_mat = copy(Dm_T)

# Define full matrix of dummies
global Dm = [Dm_J Dm_T]

# Compute the reduced matrices
Dm_tensor = reshape(Dm,TT,J,J+TT-3)
Dm_tensor_reduced = Dm_tensor[1:TT-1,1:J-1,:]
global Dm_reduced = reshape(Dm_tensor_reduced,(J-1)*(TT-1),J+TT-3)

# Define the projection matrix of the dummy matrix
global P_D = Dm_reduced*inv(Dm_reduced'*Dm_reduced)*Dm_reduced'
global P_D_OLS = inv(Dm_reduced'*Dm_reduced)*Dm_reduced'

function generate_single_mc(Pop, ξ_c, FEs, include_tau_dummy)
    # Set dimensions
    Pdims = (J = J,
             S = S,
             T = TT,
             Pop = Pop,
             FEs = FEs,
             include_tau_dummy = include_tau_dummy,
             tau_bar = tau_bar)
    
    initial_pop_dist = (Pop/(J*tau_bar))*ones(Int64,J,tau_bar)
    initial_ind_state = vcat([j*ones(Int,Int((Pop/(J*tau_bar)))) for j in 1:J*tau_bar]...)
    
    # Set Gamma tensor
    Gamma_tensor = Γ_tensor(Pdims,P_mat)
    Pgt = (Gamma_tensor = Gamma_tensor,
           initial_pop_dist = initial_pop_dist,
           initial_ind_state = initial_ind_state)
    Pdims = merge(Pdims,Pgt)

    # Set parameters based on xi type
    μ_b = 0.5; σ_b = 0.1
    μ_d = 1; σ_d = 0.5
    μ_a = 1.5; σ_a = 0.5

    if ξ_c == "exogenous"
        κ = 1; η = 1
        μ_u = 0; σ_u = 0.05
        μ_v = 0; σ_v = 0
    elseif ξ_c == "endogenous"
        κ = 0.75; η = 0.75
        μ_u = 0; σ_u = 0.05
        μ_v = 0; σ_v = 0.05
    else # zero
        κ = 1; η = 1
        μ_u = 0; σ_u = 0
        μ_v = 0; σ_v = 0
    end

    # Structure parameters
    Pparam = (μ_b = μ_b, σ_b = σ_b, μ_d = μ_d, σ_d = σ_d,
              μ_a = μ_a, σ_a = σ_a, μ_u = μ_u, σ_u = σ_u,
              μ_v = μ_v, σ_v = σ_v, α = α, β = β,
              gamma_0 = gamma_0, gamma_1 = gamma_1, gamma_2 = gamma_2,
              δ = δ, κ = κ, η = η, ρ = 0.9,
              comp_in_emp_shares = false,
              imp_prob = 10^-5, fill_prob = 10^-5,
              rmv_0_s = false, tol = 10^-10,
              max_iter = 10^5, Gamma_tensor = Gamma_tensor)

    # Generate lambdas
    d_λ_t = Normal(0, 0.1)
    λ_t = zeros(Pdims.T-1)
    d_λ_j = Normal(0, 0.1)
    λ_j = rand(d_λ_j,Pdims.J-2)
    all_true_param = [true_param; λ_j; λ_t]

    # Generate data
    D_RHS = DGP_RHS(Pdims,Pparam)
    trans_probs = gen_trans_probs(λ_t,λ_j,Pdims,Pparam,D_RHS)
    trans_probs_df = DataFrame(trans_probs,:auto)
    df_choices = empirical_choices(trans_probs,Pdims,Pparam)
    freq_probs_df = get_frequency_probs(df_choices,Pdims)

    # Create base folder structure
    folder_suffix = "J$(J)_xi$(ξ_c)_FE$(FEs)_locdummy$(include_tau_dummy)"
    
    # Save simulated data
    data_dir = joinpath(BASE_DIR, "simulated_data_full_dynamic_$(folder_suffix)")
    mkpath(data_dir)
    
    data_df = DataFrame(
        budget = D_RHS.budget,
        budget_exo = D_RHS.budget_exo,
        am_1 = D_RHS.amenities[:,1],
        am_2 = D_RHS.amenities[:,2],
        a_exo_1 = D_RHS.amenities_exo[:,1],
        a_exo_2 = D_RHS.amenities_exo[:,2]
    )
    
    CSV.write(joinpath(data_dir, "simulated_data_J$(J)_Pop$(Pop)_iter1.csv"), data_df)
    CSV.write(joinpath(data_dir, "simulated_dist_mat_J$(J)_Pop$(Pop)_iter1.csv"),
             DataFrame(D_RHS.dist_mat, :auto))

    # Save choices
    choices_dir = joinpath(BASE_DIR, "simulated_choices_full_dynamic_$(folder_suffix)")
    mkpath(choices_dir)
    CSV.write(joinpath(choices_dir, "simulated_choices_J$(J)_Pop$(Pop)_iter1.csv"), 
             df_choices)

    # Save frequency probabilities
    freq_dir = joinpath(BASE_DIR, "simulated_freq_probs_full_dynamic_$(folder_suffix)")
    mkpath(freq_dir)
    CSV.write(joinpath(freq_dir, "simulated_choices_J$(J)_Pop$(Pop)_iter1.csv"), 
             freq_probs_df)

    # Save transition probabilities
    prob_dir = joinpath(BASE_DIR, "simulated_choices_true_prob_mat_full_dynamic_$(folder_suffix)")
    mkpath(prob_dir)
    CSV.write(joinpath(prob_dir, "prob_mat_J$(J)_Pop$(Pop)_iter1.csv"), 
             trans_probs_df)

    # Save true parameters
    param_dir = joinpath(BASE_DIR, "simulated_choices_true_param_$(folder_suffix)")
    mkpath(param_dir)
    CSV.write(joinpath(param_dir, "true_params_J$(J)_Pop$(Pop)_iter1.csv"),
             DataFrame(true_params = all_true_param))
end

# Run for all combinations
for Pop in populations
    for ξ_c in xi_types
        for FEs in fixed_effects
            for tau in tau_dummy
                println("Generating data for Pop=$(Pop), xi=$(ξ_c)")
                generate_single_mc(Pop, ξ_c, FEs, tau)
            end
        end
    end
end