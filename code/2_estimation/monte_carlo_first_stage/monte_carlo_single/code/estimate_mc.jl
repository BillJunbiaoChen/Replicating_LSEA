using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Printf

# Set paths
const BASE_DIR = dirname(pwd())
const DATA_DIR = joinpath(BASE_DIR, "Data")
const OUTPUT_DIR = joinpath(BASE_DIR, "Output")

# Import helper functions - make sure this path is correct
include("MC_functions_full_dynamic_corrected_stochastic_loc_cap.jl")

# Fixed parameters
const S = 2
const TT = 10
const tau_bar = 2
const J = 25
const n_iter = 1  # Number of iterations to process

# Settings
const populations = [50000, 1000050]
const xi_types = ["endogenous", "exogenous", "zero"]

# True parameters
const α = -0.05
const β = 0.1*ones(Float64,S)
const gamma_0 = -0.0025
const gamma_1 = -0.0025
const gamma_2 = -0.5
const δ = 0.1
const true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]
const n_param = length(true_param)

function setup_dimensions(Pop)
    # Basic dimensions
    Pdims = (J = J,
             S = S,
             T = TT,
             Pop = Pop,
             FEs = true,
             include_tau_dummy = true,
             tau_bar = tau_bar)
             
    # Setup tensors
    P_mat = Matrix{Float64}([0.7 0.3; 0 1])
    Gamma_tensor = Γ_tensor(Pdims, P_mat)
    Gamma_tensor_full_t = Γ_tensor_full_t(Pdims, P_mat)
    
    return merge(Pdims, (Gamma_tensor = Gamma_tensor,
                        Gamma_tensor_full_t = Gamma_tensor_full_t,
                        max_iter = 10^5))
end

function setup_matrices(Pdims)
    # Setup dummy matrices
    global loc_dummy_mat, time_dummy_mat, Dm_reduced, P_D, P_D_OLS
    
    M_J = vcat(Matrix{Int64}(I, Pdims.J-2,Pdims.J-2),zeros(Int64,2,Pdims.J-2))
    loc_dummy_mat = repeat(M_J,inner=(Pdims.T,1))

    M_T = vcat(Matrix{Int64}(I, Pdims.T-1,Pdims.T-1),zeros(Int64,1,Pdims.T-1))
    time_dummy_mat = vcat(repeat(M_T,outer = (Pdims.J-1,1)),zeros(Pdims.T,Pdims.T-1))

    Dm = loc_dummy_mat

    # Setup projection matrices
    Dm_tensor = reshape(Dm,Pdims.T,Pdims.J,Pdims.J-2)
    Dm_tensor_reduced = Dm_tensor[1:Pdims.T-1,1:Pdims.J-1,:]
    Dm_reduced = reshape(Dm_tensor_reduced,(Pdims.J-1)*(Pdims.T-1),Pdims.J-2)
    P_D = Dm_reduced*inv(Dm_reduced'*Dm_reduced)*Dm_reduced'
    P_D_OLS = inv(Dm_reduced'*Dm_reduced)*Dm_reduced'

    return loc_dummy_mat, time_dummy_mat, Dm_reduced, P_D, P_D_OLS
end

function process_xi_type(ξ_c::String)
    for (j_id,J) in enumerate([25])
        for (pop_id,Pop) in enumerate(populations)
            println("\nProcessing: xi=$(ξ_c), Pop=$(Pop)")
            
            # Initialize storage
            L_2s = zeros(n_iter)
            L_infs = zeros(n_iter)
            bias_projected_true = zeros(n_iter,n_param)
            abs_bias_projected_true = zeros(n_iter,n_param)
            bias_projected_MNL = zeros(n_iter,n_param)
            abs_bias_projected_MNL = zeros(n_iter,n_param)
            bias_projected_freq = zeros(n_iter,n_param)
            abs_bias_projected_freq = zeros(n_iter,n_param)

            # Setup dimensions once outside the loop
            global Pdims = setup_dimensions(Pop)
            loc_dummy_mat, time_dummy_mat, Dm_reduced, P_D, P_D_OLS = setup_matrices(Pdims)
            Pparam = (ρ = 0.9, fill_prob = 10^-6)

            Threads.@threads for k in 1:n_iter
                try
                    println("#######################################################################")
                    println("Iteration number ", k)
                    println("#######################################################################")
                    
                    # Load data and probabilities
                    folder_suffix = "J$(J)_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)"
                    
                    dat = DataFrame(CSV.File(joinpath(DATA_DIR, 
                        "simulated_data_full_dynamic_$(folder_suffix)/simulated_data_J$(J)_Pop$(Pop)_iter$(k).csv")))
                    dist_mat = Matrix(DataFrame(CSV.File(joinpath(DATA_DIR,
                        "simulated_data_full_dynamic_$(folder_suffix)/simulated_dist_mat_J$(J)_Pop$(Pop)_iter$(k).csv"))))
                    
                    tp_df = DataFrame(CSV.File(joinpath(DATA_DIR,
                        "simulated_choices_true_prob_mat_full_dynamic_$(folder_suffix)/prob_mat_J$(J)_Pop$(Pop)_iter$(k).csv")))
                    MNL_df = DataFrame(CSV.File(joinpath(DATA_DIR,
                        "simulated_choices_full_dynamic_$(folder_suffix)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv")))
                    freq_probs_df = DataFrame(CSV.File(joinpath(DATA_DIR,
                        "simulated_freq_probs_full_dynamic_$(folder_suffix)/simulated_choices_J$(J)_Pop$(Pop)_iter$(k).csv")))

                    #######################################################################
                    # Evaluate MNL choice probabilities
                    #######################################################################

                    # Get the true transition probability matrix
                    trans_probs_df = parse.(Float64,string.(tp_df))
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

                    #######################################################################
                    # Estimate the model
                    #######################################################################
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

                    
                catch e
                    println("Error in iteration $k:")
                    println(e)
                    println(stacktrace())
                end
            end

            # Calculate averages
            avg_L_inf = mean(L_infs)
            avg_L_2 = mean(L_2s)
            avg_bias_true = vec(mean(bias_projected_true,dims=1))
            avg_abs_bias_true = vec(mean(abs_bias_projected_true,dims=1))
            avg_bias_MNL = vec(mean(bias_projected_MNL,dims=1))
            avg_abs_bias_MNL = vec(mean(abs_bias_projected_MNL,dims=1))
            avg_bias_freq = vec(mean(bias_projected_freq,dims=1))
            avg_abs_bias_freq = vec(mean(abs_bias_projected_freq,dims=1))

            # Save L2/Linf metrics
            df_output_MNL = DataFrame(
                Pop = [Pop],
                J = [J],
                mean_L2 = [avg_L_2],
                mean_Linf = [avg_L_inf]
            )
            
            CSV.write(joinpath(OUTPUT_DIR,
                "simulated_choices_results_25locs_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_cr.csv"),
                df_output_MNL)

            # Save bias results
            df_output_quants = DataFrame(Pop = [Pop], J = [J])
            
            df_output_quants[!,"b_alpha_true"] = [avg_bias_true[1]]
            df_output_quants[!,"b_alpha_MNL"] = [avg_bias_MNL[1]]
            df_output_quants[!,"b_alpha_freq"] = [avg_bias_freq[1]]
            df_output_quants[!,"ab_alpha_true"] = [avg_abs_bias_true[1]]
            df_output_quants[!,"ab_alpha_MNL"] = [avg_abs_bias_MNL[1]]
            df_output_quants[!,"ab_alpha_freq"] = [avg_abs_bias_freq[1]]

            for s in 1:S
                df_output_quants[!,"b_beta_$(s)_true"] = [avg_bias_true[1+s]]
                df_output_quants[!,"b_beta_$(s)_MNL"] = [avg_bias_MNL[1+s]]
                df_output_quants[!,"b_beta_$(s)_freq"] = [avg_bias_freq[1+s]]
                df_output_quants[!,"ab_beta_$(s)_true"] = [avg_abs_bias_true[1+s]]
                df_output_quants[!,"ab_beta_$(s)_MNL"] = [avg_abs_bias_MNL[1+s]]
                df_output_quants[!,"ab_beta_$(s)_freq"] = [avg_abs_bias_freq[1+s]]
            end

            df_output_quants[!,"b_gamma_0_true"] = [avg_bias_true[1+S+1]]
            df_output_quants[!,"b_gamma_0_MNL"] = [avg_bias_MNL[1+S+1]]
            df_output_quants[!,"b_gamma_0_freq"] = [avg_bias_freq[1+S+1]]
            df_output_quants[!,"ab_gamma_0_true"] = [avg_abs_bias_true[1+S+1]]
            df_output_quants[!,"ab_gamma_0_MNL"] = [avg_abs_bias_MNL[1+S+1]]
            df_output_quants[!,"ab_gamma_0_freq"] = [avg_abs_bias_freq[1+S+1]]

            df_output_quants[!,"b_gamma_1_true"] = [avg_bias_true[1+S+2]]
            df_output_quants[!,"b_gamma_1_MNL"] = [avg_bias_MNL[1+S+2]]
            df_output_quants[!,"b_gamma_1_freq"] = [avg_bias_freq[1+S+2]]
            df_output_quants[!,"ab_gamma_1_true"] = [avg_abs_bias_true[1+S+2]]
            df_output_quants[!,"ab_gamma_1_MNL"] = [avg_abs_bias_MNL[1+S+2]]
            df_output_quants[!,"ab_gamma_1_freq"] = [avg_abs_bias_freq[1+S+2]]

            df_output_quants[!,"b_gamma_2_true"] = [avg_bias_true[1+S+3]]
            df_output_quants[!,"b_gamma_2_MNL"] = [avg_bias_MNL[1+S+3]]
            df_output_quants[!,"b_gamma_2_freq"] = [avg_bias_freq[1+S+3]]
            df_output_quants[!,"ab_gamma_2_true"] = [avg_abs_bias_true[1+S+3]]
            df_output_quants[!,"ab_gamma_2_MNL"] = [avg_abs_bias_MNL[1+S+3]]
            df_output_quants[!,"ab_gamma_2_freq"] = [avg_abs_bias_freq[1+S+3]]

            df_output_quants[!,"b_delta_true"] = [avg_bias_true[1+S+4]]
            df_output_quants[!,"b_delta_MNL"] = [avg_bias_MNL[1+S+4]]
            df_output_quants[!,"b_delta_freq"] = [avg_bias_freq[1+S+4]]
            df_output_quants[!,"ab_delta_true"] = [avg_abs_bias_true[1+S+4]]
            df_output_quants[!,"ab_delta_MNL"] = [avg_abs_bias_MNL[1+S+4]]
            df_output_quants[!,"ab_delta_freq"] = [avg_abs_bias_freq[1+S+4]]

            # Save full results
            CSV.write(joinpath(OUTPUT_DIR,
                "avg_bias_experiments_MNL_full_dynamic_25locs_xi$(ξ_c)_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_cr.csv"),
                df_output_quants)
        end
    end
end

function main()
    # Create Output directory if it doesn't exist
    isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)
    
    for ξ_c in xi_types
        try
            process_xi_type(ξ_c)
        catch e
            println("Error processing xi type $ξ_c:")
            println(e)
            println(stacktrace())
        end
    end
end

# Run the estimation
main()