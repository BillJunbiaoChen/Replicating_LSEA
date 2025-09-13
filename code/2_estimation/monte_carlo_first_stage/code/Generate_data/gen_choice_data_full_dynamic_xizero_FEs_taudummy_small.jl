####################################################################################  
## Code: MC Data Generation
## Authors: Marek Bojko & Sriram Tolety 
####################################################################################  

using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Distributed

# Read data
cd("../../")
println(pwd())
include("MC_functions_full_dynamic_corrected_stochastic_loc_cap.jl")

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))
##########################################################################################
# SEED 1243, NO FIXED EFFECTS
##########################################################################################
# Fix seed for reproducibility
Random.seed!(1243)

# Compute avg bias for different parameter specifications
J_to_consider = [25]
Pops_to_consider = [50000]
ξ_to_consider = ["zero"]

# Compute the number of inner iterations (i.e. how many times do we draw a dataset?)
n_iter = 10

#Dimensions
S = 2
TT = 10
tau_bar = 2

# Transition matrix for location capital conditional on staying in the same location
P_mat = Matrix{Float64}([0.7 0.3; 0 1])

for ξ_c in ξ_to_consider
    for J in J_to_consider
        for Pop in Pops_to_consider
                println(J)

                initial_pop_dist = (Pop/(J*tau_bar))*ones(Int64,J,tau_bar)
                initial_ind_state = vcat([j*ones(Int,Int((Pop/(J*tau_bar)))) for j in 1:J*tau_bar]...)

                # Basic dimensions (immutable)
                Pdims = (J = J,
                        S = S,
                        T = TT,
                        Pop = Pop,
                        FEs = true,
                        include_tau_dummy = true,
                        tau_bar = tau_bar)
                Gamma_tensor = Γ_tensor(Pdims,P_mat)
                Pgt = (Gamma_tensor = Gamma_tensor,
                    initial_pop_dist = initial_pop_dist,
                    initial_ind_state = initial_ind_state)
                Pdims = merge(Pdims,Pgt)

                # Basic parameters for data generating process
                μ_b = 0.5;
                σ_b = 0.1;
                μ_d = 1;
                σ_d = 0.5;
                μ_a = 1.5
                σ_a = 0.5

                if ξ_c == "exogenous"
                   # Define coefficients on the convex comb of exo and endo parts of regressors
                   κ = 1
                   η = 1
                   μ_u = 0
                   σ_u = 0.05
                   μ_v = 0
                   σ_v = 0
                elseif ξ_c == "endogenous"
                   κ = 0.75
                   η = 0.75
                   μ_u = 0
                   σ_u = 0.05
                   μ_v = 0
                   σ_v = 0.05
                elseif ξ_c == "zero"
                   κ = 1
                   η = 1
                   μ_u = 0
                   σ_u = 0
                   μ_v = 0
                   σ_v = 0
                end

                # coefficients
                α = -0.05
                β = 0.1*ones(Float64,Pdims.S)
                gamma_0 = -0.0025
                gamma_1 = -0.0025
                gamma_2 = -0.5
                δ = 0.1

                true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]

                # Decide whether to use empirical shares or true shares in the computation
                comp_in_emp_shares = false

                # Structure of parameters
                Pparam = (μ_b  = μ_b,
                         σ_b  = σ_b,
                         μ_d  = μ_d,
                         σ_d  = σ_d,
                         μ_a  = μ_a,
                         σ_a  = σ_a,
                         μ_u  = μ_u,
                         σ_u  = σ_u,
                         μ_v  = μ_v,
                         σ_v  = σ_v,
                         α    = α,
                         β    = β,
                         gamma_0 = gamma_0,
                         gamma_1 = gamma_1,
                         gamma_2 = gamma_2,
                         δ    = δ,
                         κ    = κ,
                         η    = η,
                         ρ    = 0.9,
                         comp_in_emp_shares = comp_in_emp_shares,
                         imp_prob = 10^-5,
                         fill_prob = 10^-5,
                         rmv_0_s = false,
                         tol = 10^-10,
                         max_iter = 10^5,
                         Gamma_tensor = Gamma_tensor)

                # lambdas
                d_λ_t = Normal(0, 0.1)
                λ_t = zeros(Pdims.T-1)
                d_λ_j = Normal(0, 0.1)
                λ_j = rand(d_λ_j,Pdims.J-2)

                all_true_param = [true_param; λ_j; λ_t]
                n_total_param = length(all_true_param)

                # Compute the matrix of location dummies for the utility function
                M_J = vcat(Matrix{Int64}(I, Pdims.J-2,Pdims.J-2),zeros(Int64,2,Pdims.J-2))
                Dm_J = repeat(M_J,inner=(Pdims.T,1))
                global loc_dummy_mat = copy(Dm_J)

                # Compute the matrix of time dummies for the utility function
                M_T = vcat(Matrix{Int64}(I, Pdims.T-1,Pdims.T-1),zeros(Int64,1,Pdims.T-1))
                Dm_T = vcat(repeat(M_T,outer = (Pdims.J-1,1)),zeros(Pdims.T,Pdims.T-1))
                global time_dummy_mat = copy(Dm_T)

                # Define full matrix of dummies
                global Dm = [Dm_J Dm_T]

                loc_dummy_mat_noout = loc_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
                loc_dummy_mat_full = repeat(loc_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
                time_dummy_mat_noout = time_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
                time_dummy_mat_full = repeat(time_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
                dummy_mat_full = [loc_dummy_mat_full time_dummy_mat_full]
                dummy_mat_full_reduced = remove_T_mat(dummy_mat_full,Pdims)
                dummy_mat_full_reduced_inverted = invert_indices_matrix(dummy_mat_full_reduced,Pdims)

                # For the purpose of estimation, we're dropping the outside option and the last time period
                Dm_tensor = reshape(Dm,Pdims.T,Pdims.J,Pdims.J+Pdims.T-3)
                Dm_tensor_reduced = Dm_tensor[1:Pdims.T-1,1:Pdims.J-1,:]
                global Dm_reduced = reshape(Dm_tensor_reduced,(Pdims.J-1)*(Pdims.T-1),Pdims.J+Pdims.T-3)

                Dm_f = repeat(Dm_reduced,inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))

                # Define the projection matrix of the dummy matrix
                global P_D = Dm_reduced*inv(Dm_reduced'*Dm_reduced)*Dm_reduced'
                global P_D_OLS = inv(Dm_reduced'*Dm_reduced)*Dm_reduced'

                ################################################################################
                # Generate datasets, transition probabilities, and choices
                ################################################################################
                ## Iterations
                bias_projected = zeros(n_iter,n_total_param)
                abs_bias_projected = zeros(n_iter,n_total_param)

                Threads.@threads for k=1:n_iter

                    println()
                    println("####################### Iteration number ", k, " #######################")
                    println()

                    D_RHS = DGP_RHS(Pdims,Pparam)
                    trans_probs = gen_trans_probs(λ_t,λ_j,Pdims,Pparam,D_RHS)
                    trans_probs_df = DataFrame(trans_probs,:auto)
                    df_choices = empirical_choices(trans_probs,Pdims,Pparam)
                    freq_probs_df = get_frequency_probs(df_choices,Pdims)

                    budget = D_RHS.budget
                    budget_exo = D_RHS.budget_exo
                    amenities = D_RHS.amenities
                    amenities_exo = D_RHS.amenities_exo
                    dist_mat = D_RHS.dist_mat

                    # Store data
                    dat = DataFrame(budget = budget, budget_exo = budget_exo)
                    for s in 1:Pdims.S
                        colname_a = "am_$s"
                        colname_a_exo = "a_exo_$s"
                        dat[!,colname_a] = amenities[:,s]
                        dat[!,colname_a_exo] = amenities_exo[:,s]
                    end

                    # Store distance matrix
                    dist_mat_df = DataFrame(dist_mat,:auto)

                    # Store true parameters
                    true_param_df = DataFrame(true_params = all_true_param)

                    # Output
                    isdir("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/simulated_data_J$(J)_Pop$(Pop)_iter$(k).csv",  dat)
                    CSV.write("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/simulated_dist_mat_J$(J)_Pop$(Pop)_iter$(k).csv",  dist_mat_df)

                    isdir("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  df_choices)

                    isdir("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  df_choices)

                    isdir("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  freq_probs_df)

                    isdir("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/prob_mat_J$(J)_Pop$(Pop)_iter$(k).csv",  trans_probs_df)

                    isdir("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE") || mkdir("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE")
                    CSV.write("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE/true_params_J$(J)_Pop$(Pop)_iter$(k).csv", true_param_df)

                   
                end
        end
    end
end






##########################################################################################
# SEED 1243, WITH FIXED EFFECTS
##########################################################################################
# Fix seed for reproducibility
Random.seed!(1243)

# Compute avg bias for different parameter specifications
J_to_consider = [25]
Pops_to_consider = [50000]
ξ_to_consider = ["zero"]

# Compute the number of inner iterations (i.e. how many times do we draw a dataset?)
n_iter = 10

#Dimensions
S = 2
TT = 10
tau_bar = 2

# Transition matrix for location capital conditional on staying in the same location
P_mat = Matrix{Float64}([0.7 0.3; 0 1])

for ξ_c in ξ_to_consider
    for J in J_to_consider
        for Pop in Pops_to_consider
                println(J)

                initial_pop_dist = (Pop/(J*tau_bar))*ones(Int64,J,tau_bar)
                initial_ind_state = vcat([j*ones(Int,Int((Pop/(J*tau_bar)))) for j in 1:J*tau_bar]...)

                # Basic dimensions (immutable)
                Pdims = (J = J,
                        S = S,
                        T = TT,
                        Pop = Pop,
                        FEs = true,
                        include_tau_dummy = true,
                        tau_bar = tau_bar)
                Gamma_tensor = Γ_tensor(Pdims,P_mat)
                Pgt = (Gamma_tensor = Gamma_tensor,
                    initial_pop_dist = initial_pop_dist,
                    initial_ind_state = initial_ind_state)
                Pdims = merge(Pdims,Pgt)

                # Basic parameters for data generating process
                μ_b = 0.5;
                σ_b = 0.1;
                μ_d = 1;
                σ_d = 0.5;
                μ_a = 1.5
                σ_a = 0.5

                if ξ_c == "exogenous"
                   # Define coefficients on the convex comb of exo and endo parts of regressors
                   κ = 1
                   η = 1
                   μ_u = 0
                   σ_u = 0.05
                   μ_v = 0
                   σ_v = 0
                elseif ξ_c == "endogenous"
                   κ = 0.75
                   η = 0.75
                   μ_u = 0
                   σ_u = 0.05
                   μ_v = 0
                   σ_v = 0.05
                elseif ξ_c == "zero"
                   κ = 1
                   η = 1
                   μ_u = 0
                   σ_u = 0
                   μ_v = 0
                   σ_v = 0
                end

                # coefficients
                α = -0.05
                β = 0.1*ones(Float64,Pdims.S)
                gamma_0 = -0.0025
                gamma_1 = -0.0025
                gamma_2 = -0.5
                δ = 0.1

                true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]

                # Decide whether to use empirical shares or true shares in the computation
                comp_in_emp_shares = false

                # Structure of parameters
                Pparam = (μ_b  = μ_b,
                         σ_b  = σ_b,
                         μ_d  = μ_d,
                         σ_d  = σ_d,
                         μ_a  = μ_a,
                         σ_a  = σ_a,
                         μ_u  = μ_u,
                         σ_u  = σ_u,
                         μ_v  = μ_v,
                         σ_v  = σ_v,
                         α    = α,
                         β    = β,
                         gamma_0 = gamma_0,
                         gamma_1 = gamma_1,
                         gamma_2 = gamma_2,
                         δ    = δ,
                         κ    = κ,
                         η    = η,
                         ρ    = 0.9,
                         comp_in_emp_shares = comp_in_emp_shares,
                         imp_prob = 10^-5,
                         fill_prob = 10^-5,
                         rmv_0_s = false,
                         tol = 10^-10,
                         max_iter = 10^5,
                         Gamma_tensor = Gamma_tensor)

                # lambdas
                d_λ_t = Normal(0, 0.1)
                λ_t = rand(d_λ_t,Pdims.T-1)
                d_λ_j = Normal(0, 0.1)
                λ_j = rand(d_λ_j,Pdims.J-2)

                all_true_param = [true_param; λ_j; λ_t]
                n_total_param = length(all_true_param)

                # Compute the matrix of location dummies for the utility function
                M_J = vcat(Matrix{Int64}(I, Pdims.J-2,Pdims.J-2),zeros(Int64,2,Pdims.J-2))
                Dm_J = repeat(M_J,inner=(Pdims.T,1))
                global loc_dummy_mat = copy(Dm_J)

                # Compute the matrix of time dummies for the utility function
                M_T = vcat(Matrix{Int64}(I, Pdims.T-1,Pdims.T-1),zeros(Int64,1,Pdims.T-1))
                Dm_T = vcat(repeat(M_T,outer = (Pdims.J-1,1)),zeros(Pdims.T,Pdims.T-1))
                global time_dummy_mat = copy(Dm_T)

                # Define full matrix of dummies
                global Dm = [Dm_J Dm_T]

                loc_dummy_mat_noout = loc_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
                loc_dummy_mat_full = repeat(loc_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
                time_dummy_mat_noout = time_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
                time_dummy_mat_full = repeat(time_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
                dummy_mat_full = [loc_dummy_mat_full time_dummy_mat_full]
                dummy_mat_full_reduced = remove_T_mat(dummy_mat_full,Pdims)
                dummy_mat_full_reduced_inverted = invert_indices_matrix(dummy_mat_full_reduced,Pdims)

                # For the purpose of estimation, we're dropping the outside option and the last time period
                Dm_tensor = reshape(Dm,Pdims.T,Pdims.J,Pdims.J+Pdims.T-3)
                Dm_tensor_reduced = Dm_tensor[1:Pdims.T-1,1:Pdims.J-1,:]
                global Dm_reduced = reshape(Dm_tensor_reduced,(Pdims.J-1)*(Pdims.T-1),Pdims.J+Pdims.T-3)

                Dm_f = repeat(Dm_reduced,inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))

                # Define the projection matrix of the dummy matrix
                global P_D = Dm_reduced*inv(Dm_reduced'*Dm_reduced)*Dm_reduced'
                global P_D_OLS = inv(Dm_reduced'*Dm_reduced)*Dm_reduced'

                ################################################################################
                # Generate datasets, transition probabilities, and choices
                ################################################################################
                ## Iterations
                bias_projected = zeros(n_iter,n_total_param)
                abs_bias_projected = zeros(n_iter,n_total_param)

                Threads.@threads for k=1:n_iter

                    println()
                    println("####################### Iteration number ", k, " #######################")
                    println()

                    D_RHS = DGP_RHS(Pdims,Pparam)
                    trans_probs = gen_trans_probs(λ_t,λ_j,Pdims,Pparam,D_RHS)
                    trans_probs_df = DataFrame(trans_probs,:auto)
                    df_choices = empirical_choices(trans_probs,Pdims,Pparam)
                    freq_probs_df = get_frequency_probs(df_choices,Pdims)

                    budget = D_RHS.budget
                    budget_exo = D_RHS.budget_exo
                    amenities = D_RHS.amenities
                    amenities_exo = D_RHS.amenities_exo
                    dist_mat = D_RHS.dist_mat

                    # Store data
                    dat = DataFrame(budget = budget, budget_exo = budget_exo)
                    for s in 1:Pdims.S
                        colname_a = "am_$s"
                        colname_a_exo = "a_exo_$s"
                        dat[!,colname_a] = amenities[:,s]
                        dat[!,colname_a_exo] = amenities_exo[:,s]
                    end

                    # Store distance matrix
                    dist_mat_df = DataFrame(dist_mat,:auto)

                    # Store true parameters
                    true_param_df = DataFrame(true_params = all_true_param)

                    # Output
                    isdir("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_data_J$(J)_Pop$(Pop)_iter$(k).csv",  dat)
                    CSV.write("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_dist_mat_J$(J)_Pop$(Pop)_iter$(k).csv",  dist_mat_df)

                    isdir("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  df_choices)

                    isdir("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_raw_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  df_choices)

                    isdir("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv",  freq_probs_df)

                    isdir("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/prob_mat_J$(J)_Pop$(Pop)_iter$(k).csv",  trans_probs_df)

                    isdir("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)") || mkdir("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)")
                    CSV.write("Data/simulated_choices_true_param_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/true_params_J$(J)_Pop$(Pop)_iter$(k).csv", true_param_df)

                end
        end
    end
end
