####################################################################################  
## Code: MC Estimation
## Authors: Marek Bojko & Sriram Tolety 
####################################################################################  
using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Plots, Distributed

cd("../../")
include("MC_functions_full_dynamic_corrected_stochastic_loc_cap.jl")

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Fix seed for reproducibility
Random.seed!(1243)

# Compute avg bias for different parameter specifications
J_to_consider = [25]
Pops_to_consider = [50000,1000050]
ξ_to_consider = ["exogenous"]

# Compute the number of inner iterations (i.e. how many times do we draw a dataset?)
n_iter = 10

#Dimensions
S = 2
TT = 10
tau_bar = 2

Pdims = (S = S,
         T = TT,
         FEs = true,
         include_tau_dummy = true,
         tau_bar = tau_bar,
         divide_by_params = true)

# For estimation, define true parameters
α = -0.05
β = 0.1*ones(Float64,Pdims.S)
#γ = -0.0025
gamma_0 = -0.0025
gamma_1 = -0.0025
gamma_2 = -0.5
δ = 0.1

true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]

n_param = length(true_param)

# Transition matrix for location capital conditional on staying in the same location
P_mat = Matrix{Float64}([0.7 0.3; 0 1])

# Prepare objects to store computed values
avg_L_inf = zeros(length(J_to_consider),length(Pops_to_consider))
avg_L_2 = zeros(length(J_to_consider),length(Pops_to_consider))
avg_bias_true = zeros(length(J_to_consider),length(Pops_to_consider), n_param)
avg_abs_bias_true = zeros(length(J_to_consider),length(Pops_to_consider), n_param)
avg_bias_MNL = zeros(length(J_to_consider),length(Pops_to_consider), n_param)
avg_abs_bias_MNL = zeros(length(J_to_consider),length(Pops_to_consider), n_param)
avg_bias_freq = zeros(length(J_to_consider),length(Pops_to_consider), n_param)
avg_abs_bias_freq = zeros(length(J_to_consider),length(Pops_to_consider), n_param)

# Loop through the files
for ξ_c in ξ_to_consider
   for (j_id,J) in enumerate(J_to_consider)
      for (pop_id,Pop) in enumerate(Pops_to_consider)
         L_2s = zeros(n_iter)
         L_infs = zeros(n_iter)
         bias_projected_true = zeros(n_iter,n_param)
         abs_bias_projected_true = zeros(n_iter,n_param)
         bias_projected_MNL = zeros(n_iter,n_param)
         abs_bias_projected_MNL = zeros(n_iter,n_param)
         bias_projected_freq = zeros(n_iter,n_param)
         abs_bias_projected_freq = zeros(n_iter,n_param)

         Threads.@threads for k in 1:n_iter

            println("#######################################################################")
            println(k)
            println("#######################################################################")

            Pgtw = (J = J,
                  Pop = Pop)
            global Pdims = merge(Pdims,Pgtw)
            # Basic dimensions (immutable)
            Gamma_tensor = Γ_tensor(Pdims,P_mat)
            Gamma_tensor_full_t = Γ_tensor_full_t(Pdims,P_mat)
            Pgt = (Gamma_tensor = Gamma_tensor,
                  Gamma_tensor_full_t = Gamma_tensor_full_t,
                  max_iter = 10^5)
            global Pdims = merge(Pdims,Pgt)

            Pparam = (ρ = 0.9,
                     fill_prob = 10^-6)

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

            #######################################################################
            # Evaluate MNL choice probabilities
            #######################################################################

            # Get the true transition probability matrix
            trans_probs_df = DataFrame(CSV.File("Data/simulated_choices_true_prob_mat_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/prob_mat_J$(J)_Pop$(Pop)_iter$(k).csv"))
            trans_probs_df = parse.(Float64,string.(trans_probs_df))
            trans_probs_mat = Matrix(trans_probs_df)
            trans_probs_locs = compress_mat_loc_capital(trans_probs_mat,Pdims)
            trans_probs_mnl_compatible = reshape_mnl_compatible(trans_probs_locs,Pdims)

            # Transform back to a dataframe
            tp_mnl_df = DataFrame(trans_probs_mnl_compatible,:auto)
            indices_all = [(j_x,tau_x,t_x) for j_x in 1:Pdims.J for tau_x in 1:Pdims.tau_bar for t_x in 1:Pdims.T]
            tp_mnl_df[!,"t"] = [indic[3] for indic in indices_all]
            tp_mnl_df[!,"tau_init"] = [indic[2] for indic in indices_all]
            tp_mnl_df[!,"j_init"] = [indic[1] for indic in indices_all]

            # Get the predicted CCPs from the MNL
            MNL_df = DataFrame(CSV.File("Data/simulated_choices_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv"))
            MNL_df[!,r"p"] = parse.(Float64,string.(MNL_df[!,r"p"]))
            if "tau_init_squared" in names(MNL_df)
               select!(MNL_df, Not(:tau_init_squared))
            end
            MNL_df_indices = MNL_df[!,["t","tau_init","j_init"]]
            # Convert into a nested list
            M_indices = Matrix(MNL_df_indices)

            # Merge the two dataframes
            joined_df = innerjoin(MNL_df, tp_mnl_df, on = [:t,:j_init,:tau_init])

            # Retrieve elements
            trans_probs_reduced = Matrix(joined_df[!,r"x"])
            MNL_probs = Matrix(joined_df[!,r"p"])

            trans_probs_reduced_vec = reshape(trans_probs_reduced,size(trans_probs_reduced)[1]*size(trans_probs_reduced)[2])
            MNL_probs_vec = reshape(MNL_probs,size(trans_probs_reduced)[1]*size(trans_probs_reduced)[2])

            replace!(x -> typeof(x) <: AbstractString ? parse(Float64,x) : x, trans_probs_reduced_vec)
            replace!(x -> typeof(x) <: AbstractString ? parse(Float64,x) : x, MNL_probs_vec)
            # Compute distances
            L_infs[k] = norm(trans_probs_reduced_vec - MNL_probs_vec,Inf)
            L_2s[k] = norm(trans_probs_reduced_vec - MNL_probs_vec,2)

            # Get transition probabilities based on frequencies
            freq_probs_df = DataFrame(CSV.File("Data/simulated_freq_probs_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv"))

            #######################################################################
            # Estimate the model
            #######################################################################

            # Retrieve data
            dat = DataFrame(CSV.File("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_data_J$(J)_Pop$(Pop)_iter$(k).csv"))
            dist_mat = Matrix(DataFrame(CSV.File("Data/simulated_data_full_dynamic_J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)/simulated_dist_mat_J$(J)_Pop$(Pop)_iter$(k).csv")))

            for j in 1:Pdims.J
               rename!(tp_mnl_df,"x$j" => "p$j")
               rename!(freq_probs_df,"x$j" => "p$j")
            end

            #### Estimate using the true probabilities
            D_true = prepare_data(dat,tp_mnl_df,dist_mat,Pdims,Pparam)
            x_init_small = rand(length(true_param))
            W =  inv(omega_mat(x_init_small,P_D,D_true,Pdims))
            fun_gmm_true(x) = gmm(x,W,P_D,D_true,Pdims)
            println(fun_gmm_true(true_param))

            @show results_projected = optimize(fun_gmm_true,
                         x_init_small,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))

            println("Step 1: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Optimal GMM
            optimal_W = inv(omega_mat(results_projected.minimizer,P_D,D_true,Pdims))

            fun_gmm_step2_true(x) = gmm(x,optimal_W,P_D,D_true,Pdims)
            println(fun_gmm_step2_true(true_param))

            @show results_projected = optimize(fun_gmm_step2_true,
                         results_projected.minimizer,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))


            println("Step 2: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Print all parameters
            @show θ_hat_true = results_projected.minimizer

            # Compute bias
            bias_projected_true[k,:] = θ_hat_true - true_param
            abs_bias_projected_true[k,:] = abs.(θ_hat_true - true_param)


            #### Estimate using the MNL
            D_MNL = prepare_data(dat,MNL_df,dist_mat,Pdims,Pparam)
            x_init_small = rand(length(true_param))
            W =  inv(omega_mat(x_init_small,P_D,D_MNL,Pdims))
            fun_gmm(x) = gmm(x,W,P_D,D_MNL,Pdims)
            println(fun_gmm(true_param))

            @show results_projected = optimize(fun_gmm,
                         x_init_small,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))

            println("Step 1: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Optimal GMM
            optimal_W = inv(omega_mat(results_projected.minimizer,P_D,D_MNL,Pdims))

            fun_gmm_step2(x) = gmm(x,optimal_W,P_D,D_MNL,Pdims)
            println(fun_gmm_step2(true_param))

            @show results_projected = optimize(fun_gmm_step2,
                         results_projected.minimizer,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))


            println("Step 2: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Print all parameters
            @show θ_hat_MNL = results_projected.minimizer

            # Compute bias
            bias_projected_MNL[k,:] = θ_hat_MNL - true_param
            abs_bias_projected_MNL[k,:] = abs.(θ_hat_MNL - true_param)

            #### Estimate using the frequency probabilities
            D_freq = prepare_data(dat,freq_probs_df,dist_mat,Pdims,Pparam)
            x_init_small = rand(length(true_param))
            W_freq =  inv(omega_mat(x_init_small,P_D,D_freq,Pdims))
            fun_gmm_freq(x) = gmm(x,W,P_D,D_freq,Pdims)
            println(fun_gmm_freq(true_param))

            @show results_projected_freq = optimize(fun_gmm_freq,
                         x_init_small,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))

            println("Step 1: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Optimal GMM
            optimal_W = inv(omega_mat(results_projected_freq.minimizer,P_D,D_freq,Pdims))

            fun_gmm_step2_freq(x) = gmm(x,optimal_W,P_D,D_freq,Pdims)
            println(fun_gmm_step2_freq(true_param))

            @show results_projected_freq = optimize(fun_gmm_step2_freq,
                         results_projected.minimizer,
                         LBFGS(),
                         autodiff = :forward,
                         Optim.Options(f_tol = 1e-16,
                                       x_tol = 1e-16,
                                       g_tol = 1e-16,
                                       show_trace = true,
                                       show_every = 1000,
                                       iterations = 10^6,
                                       allow_f_increases = true,
                                       outer_iterations = 1000))


            println("Step 2: Reached maximum of iterations:  ", results_projected.iteration_converged,", after ", results_projected.time_run, " seconds")

            # Print all parameters
            @show θ_hat_freq = results_projected_freq.minimizer

            # Compute bias
            bias_projected_freq[k,:] = θ_hat_freq - true_param
            abs_bias_projected_freq[k,:] = abs.(θ_hat_freq - true_param)

         end
         avg_L_inf[j_id,pop_id] = mean(L_infs)
         avg_L_2[j_id,pop_id] = mean(L_2s)
         avg_bias_true[j_id,pop_id,:] = vec(mean(bias_projected_true,dims=1))
         avg_abs_bias_true[j_id,pop_id,:] = vec(mean(abs_bias_projected_true,dims=1))
         avg_bias_MNL[j_id,pop_id,:] = vec(mean(bias_projected_MNL,dims=1))
         avg_abs_bias_MNL[j_id,pop_id,:] = vec(mean(abs_bias_projected_MNL,dims=1))
         avg_bias_freq[j_id,pop_id,:] = vec(mean(bias_projected_freq,dims=1))
         avg_abs_bias_freq[j_id,pop_id,:] = vec(mean(abs_bias_projected_freq,dims=1))
      end
   end

   # Output
   avg_L_inf_vec = reshape(avg_L_inf,length(J_to_consider)*length(Pops_to_consider))
   avg_L_2_vec = reshape(avg_L_2,length(J_to_consider)*length(Pops_to_consider))
   avg_bias_true_mat = reshape(avg_bias_true,length(J_to_consider)*length(Pops_to_consider),n_param)
   avg_abs_bias_true_mat = reshape(avg_abs_bias_true,length(J_to_consider)*length(Pops_to_consider),n_param)
   avg_bias_MNL_mat = reshape(avg_bias_MNL,length(J_to_consider)*length(Pops_to_consider),n_param)
   avg_abs_bias_MNL_mat = reshape(avg_abs_bias_MNL,length(J_to_consider)*length(Pops_to_consider),n_param)
   avg_bias_freq_mat = reshape(avg_bias_freq,length(J_to_consider)*length(Pops_to_consider),n_param)
   avg_abs_bias_freq_mat = reshape(avg_abs_bias_freq,length(J_to_consider)*length(Pops_to_consider),n_param)
   indices = [(Pop_x,j_x) for Pop_x in Pops_to_consider for j_x in J_to_consider]

   df_output_MNL = DataFrame(Pop = [indc[1] for indc in indices],
                           J = [indc[2] for indc in indices],
                           mean_L2 = avg_L_2_vec,
                           mean_Linf = avg_L_inf_vec)
   CSV.write("Output/simulated_choices_results_25locs_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_cr.csv",  df_output_MNL)

   df_output_quants = DataFrame(Pop = [indc[1] for indc in indices],
                           J = [indc[2] for indc in indices])
   df_output_quants[!,"b_alpha_true"] = avg_bias_true_mat[:,1]
   df_output_quants[!,"b_alpha_MNL"] = avg_bias_MNL_mat[:,1]
   df_output_quants[!,"b_alpha_freq"] = avg_bias_freq_mat[:,1]
   df_output_quants[!,"ab_alpha_true"] = avg_abs_bias_true_mat[:,1]
   df_output_quants[!,"ab_alpha_MNL"] = avg_abs_bias_MNL_mat[:,1]
   df_output_quants[!,"ab_alpha_freq"] = avg_abs_bias_freq_mat[:,1]

   for s in 1:Pdims.S
        df_output_quants[!,"b_beta_$(s)_true"] = avg_bias_true_mat[:,1+s]
        df_output_quants[!,"b_beta_$(s)_MNL"] = avg_bias_MNL_mat[:,1+s]
        df_output_quants[!,"b_beta_$(s)_freq"] = avg_bias_freq_mat[:,1+s]
        df_output_quants[!,"ab_beta_$(s)_true"] = avg_abs_bias_true_mat[:,1+s]
        df_output_quants[!,"ab_beta_$(s)_MNL"] = avg_abs_bias_MNL_mat[:,1+s]
        df_output_quants[!,"ab_beta_$(s)_freq"] = avg_abs_bias_freq_mat[:,1+s]
   end

   df_output_quants[!,"b_gamma_0_true"] = avg_bias_true_mat[:,1+Pdims.S+1]
   df_output_quants[!,"b_gamma_0_MNL"] = avg_bias_MNL_mat[:,1+Pdims.S+1]
   df_output_quants[!,"b_gamma_0_freq"] = avg_bias_freq_mat[:,1+Pdims.S+1]
   df_output_quants[!,"ab_gamma_0_true"] = avg_abs_bias_true_mat[:,1+Pdims.S+1]
   df_output_quants[!,"ab_gamma_0_MNL"] = avg_abs_bias_MNL_mat[:,1+Pdims.S+1]
   df_output_quants[!,"ab_gamma_0_freq"] = avg_abs_bias_freq_mat[:,1+Pdims.S+1]

   df_output_quants[!,"b_gamma_1_true"] = avg_bias_true_mat[:,1+Pdims.S+2]
   df_output_quants[!,"b_gamma_1_MNL"] = avg_bias_MNL_mat[:,1+Pdims.S+2]
   df_output_quants[!,"b_gamma_1_freq"] = avg_bias_freq_mat[:,1+Pdims.S+2]
   df_output_quants[!,"ab_gamma_1_true"] = avg_abs_bias_true_mat[:,1+Pdims.S+2]
   df_output_quants[!,"ab_gamma_1_MNL"] = avg_abs_bias_MNL_mat[:,1+Pdims.S+2]
   df_output_quants[!,"ab_gamma_1_freq"] = avg_abs_bias_freq_mat[:,1+Pdims.S+2]

   df_output_quants[!,"b_gamma_2_true"] = avg_bias_true_mat[:,1+Pdims.S+3]
   df_output_quants[!,"b_gamma_2_MNL"] = avg_bias_MNL_mat[:,1+Pdims.S+3]
   df_output_quants[!,"b_gamma_2_freq"] = avg_bias_freq_mat[:,1+Pdims.S+3]
   df_output_quants[!,"ab_gamma_2_true"] = avg_abs_bias_true_mat[:,1+Pdims.S+3]
   df_output_quants[!,"ab_gamma_2_MNL"] = avg_abs_bias_MNL_mat[:,1+Pdims.S+3]
   df_output_quants[!,"ab_gamma_2_freq"] = avg_abs_bias_freq_mat[:,1+Pdims.S+3]

   df_output_quants[!,"b_delta_true"] = avg_bias_true_mat[:,1+Pdims.S+4]
   df_output_quants[!,"b_delta_MNL"] = avg_bias_MNL_mat[:,1+Pdims.S+4]
   df_output_quants[!,"b_delta_freq"] = avg_bias_freq_mat[:,1+Pdims.S+4]
   df_output_quants[!,"ab_delta_true"] = avg_abs_bias_true_mat[:,1+Pdims.S+4]
   df_output_quants[!,"ab_delta_MNL"] = avg_abs_bias_MNL_mat[:,1+Pdims.S+4]
   df_output_quants[!,"ab_delta_freq"] = avg_abs_bias_freq_mat[:,1+Pdims.S+4]

   # Output to CSV
   CSV.write("Output/avg_bias_experiments_MNL_full_dynamic_25locs_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_cr.csv",  df_output_quants)
end

