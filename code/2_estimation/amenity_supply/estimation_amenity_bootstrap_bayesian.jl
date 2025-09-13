####################################################################################  
## Code: Amenity supply estimation
## Author: Milena almagro
####################################################################################  


using Distributed
@everywhere using Optim 
@everywhere using Random
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using ForwardDiff
@everywhere using CSV
@everywhere using DataFrames
@everywhere using DelimitedFiles
@everywhere using Printf
@everywhere using StatsBase
@everywhere using XLSX
@everywhere using Base.Threads
import Base.Threads.@threads
Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f)

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Set path
path = @__DIR__
cd(path)
cd("../../..")
main_path = pwd()


# Include functions
cd(main_path*"/code/2_estimation/amenity_supply/")
@everywhere include("estimation_functions.jl")

# Fix seed for reproducibility
Random.seed!(123)

# Define data structure
@everywhere mutable struct Data_struct
        Y::Vector{Float64}
        X::Matrix{Float64}
        D::Matrix{Float64}
        Z_full::Matrix{Float64}
        A::Vector{Float64}
end

####################################################################################  
## Main estimation
####################################################################################  
time_FE = 1
location_FE = 1

function driver(gamma::Float64, norm::Int64, conduct_robustness::Bool)
    display(gamma)
    γ = gamma
    norm = norm

    display("RUNNING WITH GAMMA = "*string(round(-γ, digits = 2)))

    # Read data
    cd(main_path*"/data/constructed/")
    df = XLSX.readtable("panel_amenities_structural_estimation.xlsx", "Sheet1")|>DataFrame
    Y = Vector{Float64}(df.amenity)
    A = Vector{Float64}(df.total_amenity)
    X = Matrix{Float64}(df[!,r"budget_h_"])./norm;
    Z = Matrix{Float64}(df[!,r"z_b_"]);

    # Basic dimensions
    global N = size(Y)[1]
    D_j =  Matrix{Float64}(df[!,r"dummy_j"]);
    global J = size(D_j)[2]
    D_t =  Matrix{Float64}(df[!,r"dummy_year"]);
    global T = size(D_t)[2]
    global S = 6

    # Dimensions
    global Pdims = (J = J,
            T = T,
            S = S,
            K = 7);
    global num_alpha = (Pdims.K)*Pdims.S;

    ## Choose model

    if time_FE == 1
            D = D_t
            if location_FE == 1
                    D = [D_j D[:,1:end-1]]
            end
    else
            if location_FE == 1
                    D = D_j
            else
                    D =Array{Float64}(undef, 0, 0)
            end
    end

    if time_FE + location_FE  > 0
            Z_full = [Z D]
    else
            Z_full = Z
    end

    # Create data structure
    data = Data_struct(Y,X,D,Z_full,A);

    ## Initialize
    if time_FE + location_FE  > 0
            x_init = ones(num_alpha+size(D)[2]-1)
    else
            x_init = ones(num_alpha)
    end

    p = length(x_init)

    # tolerance
    tol = 1e-15;

    # Weight matrix
    W = inv(Z_full'*Z_full/N);

    ## Define data
    global data = Data_struct(Y,X,D,Z_full,A);

    # Objective functions
    # Include functions
    fun_gmm_logs(x) = gmm_logs(x,W,data,Pdims, γ);

    ## Constrained optimization
    global x_init = 0.5*rand(Uniform(0,1),p)
    global lower = [zeros(num_alpha); -Inf*ones(p-num_alpha)]
    global upper = Inf*ones(p)
    
    # Logs
    @time results_gmm_box_log = optimize(fun_gmm_logs,
                        lower,
                        upper,
                        x_init,
                        Fminbox(BFGS()),
                        autodiff = :forward,
                        Optim.Options(f_tol = tol,
                                        x_tol = tol,
                                        g_tol = tol,
                                        show_trace = true,
                                        show_every = 10000,
                                        iterations = 10000,
                                        allow_f_increases = true,
                                        outer_iterations = 10));

    a_mat = zeros(p,1)
    a_mat[:,1] = results_gmm_box_log.minimizer
    a_mat = DataFrame(a_mat,:auto)

    cd(main_path*"/output/estimates/")
    CSV.write("amenity_supply_estimation_results_gamma_"*string(round(-γ, digits = 2))*".csv",a_mat)
    resid = log.(Y) - predict_logs(results_gmm_box_log.minimizer,data,Pdims, γ)
    resid_mat = zeros(N,6)
    resid_mat[:,1] = df.s
    resid_mat[:,2] = df.gb
    resid_mat[:,3] = df.year
    resid_mat[:,4] = resid
    resid_mat[:,5] = norm*ones(N)
    resid_mat[:,6] = predict_logs(results_gmm_box_log.minimizer,data,Pdims, γ)
    resid_mat = DataFrame(resid_mat,:auto)
    cd(main_path*"/output/estimates/")
    CSV.write("amenity_supply_residuals_yhat_gamma_"*string(round(-γ, digits =2))*".csv",resid_mat)

    cd(main_path*"/data/final/estimates/")
    CSV.write("amenity_supply_estimation_results_gamma_"*string(round(-γ, digits = 2))*".csv",a_mat)
    CSV.write("amenity_supply_residuals_yhat_gamma_"*string(round(-γ, digits = 2))*".csv",resid_mat)

    ####################################################################################  
    ## Bootstrap SE
    ####################################################################################  

    
    B = 500
    global f_w = Dirichlet(N,0.5)
    global x_init = 0.5*rand(Uniform(0,1),p)
    weight_draw_mat = zeros(B,N)
    sol_mat_W_log = zeros(B,p)
    
    Threads.@threads for i in 1:B
            println("Iteration number ", i)
            w_draws = rand(f_w,1);
            weight_draw_mat[i,:] = w_draws

            W_b = inv((w_draws.*data.Z_full)'*(w_draws.*data.Z_full)/N);
            fun_gmm_logs_b_W(x) = gmm_logs_bayesian(x,W_b,data,Pdims,w_draws, γ);
            @time results_gmm_loop_W = optimize(fun_gmm_logs_b_W,
                            lower,
                            upper,
                            x_init,
                            Fminbox(BFGS()),
                            autodiff = :forward,
                            Optim.Options(f_tol = tol,
                                        x_tol = tol,
                                        g_tol = tol,
                                        show_trace = false,
                                        show_every = 10000,
                                        iterations = 10000,
                                        allow_f_increases = true,
                                        outer_iterations = 10));

            sol_mat_W_log[i,:] = results_gmm_loop_W.minimizer;

            println("Iteration number ", i,": Nonparametric bootsrap in logs solved with ", results_gmm_loop_W.iterations, " iterations and minimum ", results_gmm_loop_W.minimum)

    end
    

    # Save matrix of results
    cd(main_path*"/output/estimates/")
    sol_mat_W_log_df = DataFrame(sol_mat_W_log,:auto)
    CSV.write("B_matrix_amenity_supply_gamma_"*string(round(-γ, digits = 2))*".csv",sol_mat_W_log_df)

    # mean
    cd(main_path*"/output/estimates/")
    sol_mat_W_log_df = DataFrame(CSV.File("B_matrix_amenity_supply_gamma_"*string(round(-γ, digits = 2))*".csv"))
    sol_mat_W_log = Matrix{Float64}(sol_mat_W_log_df)
    mean_log = vec(mapslices(mean, sol_mat_W_log; dims=1))
    mean_log_mat_box = reshape(mean_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);

    # Compute residuals
    resid = log.(Y) - predict_logs(mean_log,data,Pdims, γ)
    SSR_main = sum(resid.^2)
    resid_mat = zeros(N,5)
    resid_mat[:,1] = df.s
    resid_mat[:,2] = df.gb
    resid_mat[:,3] = df.year
    resid_mat[:,4] = resid
    resid_mat[:,5] = norm*ones(N)
    resid_mat = DataFrame(resid_mat,:auto)

    # Save mean results in simulation folder
    cd(main_path*"/data/final/estimates")
    mean_log_mat = zeros(p,1)
    mean_log_mat[:,1] = mean_log
    mean_log_mat = DataFrame(mean_log_mat,:auto)
    CSV.write("B_amenity_supply_estimation_gamma_"*string(round(-γ, digits = 2))*".csv",mean_log_mat)
    CSV.write("B_amenity_supply_residuals_gamma_"*string(round(-γ, digits = 2))*".csv",resid_mat)

    # Save results in output folder
    cd(main_path*"/output/estimates/")
    CSV.write("B_amenity_supply_estimation_gamma_"*string(round(-γ, digits = 2))*".csv",mean_log_mat)
    CSV.write("B_amenity_supply_residuals_gamma_"*string(round(-γ, digits = 2))*".csv",resid_mat)

    # 10% significance
    cd(main_path*"/output/estimates/")
    top_percentile(x) = percentile(x, 95)
    bottom_percentile(x) = percentile(x, 5)
    ub_log = mapslices(top_percentile, sol_mat_W_log; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_90 = [lb_log; ub_log]
    CI_log_90 = DataFrame(CI_log_90,:auto)
    CSV.write("B_CI90_gamma_"*string(round(-γ, digits = 2))*".csv",CI_log_90)

    # 5% significance
    top_percentile(x) = percentile(x, 97.5)
    bottom_percentile(x) = percentile(x, 2.5)
    ub_log = mapslices(top_percentile, sol_mat_W_log; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_95 = [lb_log; ub_log]
    CI_log_95 = DataFrame(CI_log_95,:auto)
    CSV.write("B_CI95_gamma_"*string(round(-γ, digits = 2))*".csv",CI_log_95)


    # 1% significance
    top_percentile(x) = percentile(x, 99.5)
    bottom_percentile(x) = percentile(x, 0.5)
    ub_log = mapslices(top_percentile, sol_mat_W_log; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_99 = [lb_log; ub_log]
    # Save matrix of results
    CI_log_99 = DataFrame(CI_log_99,:auto)
    CSV.write("B_CI99_gamma_"*string(round(-γ, digits = 2))*".csv",CI_log_99)


    
    ####################################################################################  
    ## Compose latex table
    ####################################################################################  

    # Compute the number of stars based on significance levels
    # Total number of parameters
    num_params = Pdims.K * Pdims.S  # Should be 7 * 6 = 42

    # Ensure that CI matrices have the correct size
    CI_log_90 = CI_log_90[:, 1:num_params]
    CI_log_90 = Matrix(CI_log_90)
    
    CI_log_95 = CI_log_95[:, 1:num_params]
    CI_log_95 = Matrix(CI_log_95)

    CI_log_99 = CI_log_99[:, 1:num_params]
    CI_log_99 = Matrix(CI_log_99)


    # Reshape the data
    CI90_lb = reshape(CI_log_90[1, :], Pdims.K, Pdims.S)

    CI95_lb = reshape(CI_log_95[1, :], Pdims.K, Pdims.S)

    CI99_lb = reshape(CI_log_99[1, :], Pdims.K, Pdims.S)

    epsilon = 1e-3
    stars = zeros(Int,Pdims.K,Pdims.S)
    # Determine the number of stars for each estimate
    for i in 1:Pdims.K
        for j in 1:Pdims.S
            if CI99_lb[i,j] > epsilon
                stars[i,j] = 3  # Significant at 1%
            elseif CI95_lb[i,j] > epsilon
                stars[i,j] = 2  # Significant at 5%
            elseif CI90_lb[i,j] > epsilon
                stars[i,j] = 1  # Significant at 10%
            else
                stars[i,j] = 0  # Not significant
            end
        end
    end

    display(stars)


    df_output = Matrix{Float64}(CI_log_95[:,1:6*7])
    df_output_lb = vec(df_output[1,:])
    r_df_output_lb = reshape(df_output_lb,7,6)
    df_output_ub = vec(df_output[2,:])
    r_df_output_ub = reshape(df_output_ub,7,6)

    # Attach names
    group_names = ["Older Families";"Singles"; "Younger Families" ;"Students" ;"Immigrant Families" ;"Dutch Low Income" ;"Tourists"]
    r_df_output = hcat(group_names,mean_log_mat_box)
    amenity_names = ["Touristic Amenities" "Restaurants" "Bars" "Food Stores" "Non-Food Stores" "Nurseries"]
    amenity_string = ""
    for i in 1:6
    amenity_string *= "&"*amenity_names[i]
    end

    # Generate table header
    Table  = "\\begin{table}[h!]\n";
    Table *= "\\setlength{\\tabcolsep}{1pt}\n";
    Table *= "\\centering \n";
    Table *= "\\caption{Estimates of amenity supply parameters.}\n";
    Table *= "\\label{tab:amenity_supply_estimation}\n";

    Table *= "\\scalebox{0.65}{\\begin{tabular}{lcccccc}\n";
    Table *= "    \\toprule\n";

    Table *= "    % Table header\n";
    Table *=  amenity_string*"\\\\\n";
    Table *= "    \\midrule\n";
    # Generate table body
    Table *= "    % Table body\n";
    for row in 1:7
        Table *= r_df_output[row,1]
        for column in 2:7
            estimate = round(r_df_output[row,column],digits=3)
            num_stars = stars[row, column-1]
            star_string = "*"^num_stars
            Table *= " & " * string(estimate) * star_string
        end
        Table *= "\\\\\n";
        for col in 1:6
            Table *= " & [" * string(round(r_df_output_lb[row,col],digits=3)) *","*string(round(r_df_output_ub[row,col],digits=3)) *"]"
        end
        Table *= "\\\\\n";
    end
    Table *= "  \\bottomrule\n";
    Table *= "\\end{tabular}}\n";
    Table *= """\\legend{This table reports bootstrap results for coefficients \$\\beta^k_s\$ from Equation \\ref{eq:amenities_regression} using a three-way panel of 22 districts in Amsterdam for 2008-2018 over 500 draws. Parameters \$\\beta^k_s\$ and fixed effects \$\\lambda_j\$ and \$\\lambda_t\$ are estimated via GMM, where we restrict parameters to be weakly positive as implied by the microfoundation of the amenity model in Appendix \\ref{oa-sec: appendix microfoundation-amenity demand}. The estimation procedure is outlined in section \\ref{sec: amenity_supply estimation} following a Bayesian-bootstrap with random Dirichlet weights. Total expenditure \$X^k_{jt}\$ is measured in thousands of Euros. Top rows indicate average estimates of the bootstrap samples. Results inside square brackets indicate 95\\% confidence intervals. We omit estimates of the location and time fixed effects. \\sym{*}\\(p<0.10\\), \\sym{**}\\(p<0.05\\), \\sym{***}\\(p<0.01\\).}"""
    Table *= "\\end{table}\n";

    # Export result to .tex file
    cd(main_path*"/output/tables/")
    write("table_output_amenity_supply_gamma_"*string(round(-γ, digits = 2))*".tex", Table);


    ####################################################################################  
    ## Economic significance
    ####################################################################################  


    cd(main_path*"/output/estimates/")
    sol_mat_W_log = Matrix(DataFrame(CSV.File("B_matrix_amenity_supply_gamma_"*string(round(-γ, digits = 2))*".csv")))
    mean_log = vec(mapslices(mean, sol_mat_W_log; dims=1))

    observed_levels = predict_levels(mean_log,data,Pdims, γ)
    CF_X = Matrix{Float64}(df[!,r"budget_h_"])./norm;
    for i in 1:6
            CF_X[:,7+(i-1)*6] = 1.1*CF_X[:,7+(i-1)*6]
    end


    CF_data = data;
    CF_data.X = CF_X;
    CF_levels = predict_levels(mean_log,CF_data,Pdims, γ)

    CF_growth = CF_levels./observed_levels

    CF_growth_touristic = CF_growth[df.s .== 1]
    CF_growth_restaurants = CF_growth[df.s .== 2]
    CF_growth_bars= CF_growth[df.s .== 3]
    CF_growth_food = CF_growth[df.s .== 4]
    CF_growth_non_food = CF_growth[df.s .== 5]
    CF_growth_nurseries = CF_growth[df.s .== 6]

    economic_significance_results = [mean(CF_growth) mean(CF_growth_touristic) mean(CF_growth_restaurants) mean(CF_growth_bars) mean(CF_growth_food) mean(CF_growth_non_food) mean(CF_growth_nurseries)]

    # Save results 
    cd(main_path*"/output/estimates/")
    economic_significance_results = DataFrame(economic_significance_results,:auto)
    CSV.write("Economic_significance_gamma_"*string(round(-γ, digits = 2))*".csv",economic_significance_results)

    if conduct_robustness == false
        return
    end

    ####################################################################################  
    ## Robustness check to regulations
    ####################################################################################  
    
    D = [D_j Matrix{Float64}(df[!,r"int"])]
    Z_full = [Z D]
    W = inv(Z_full'*Z_full/N);
    x_init = ones(num_alpha+size(D)[2]-1)
    p = length(x_init)

    data = Data_struct(Y,X,D,Z_full,A);

    fun_gmm_logs(x) = gmm_logs(x,W,data,Pdims, γ);

    lower = [zeros(num_alpha); -Inf*ones(p-num_alpha)]
    upper = Inf*ones(p)

    # Logs
    results_gmm_box_log = optimize(fun_gmm_logs,
                        lower,
                        upper,
                        x_init,
                        Fminbox(BFGS()),
                        autodiff = :forward,
                        Optim.Options(f_tol = tol,
                                        x_tol = tol,
                                        g_tol = tol,
                                        show_trace = true,
                                        show_every = 10000,
                                        iterations = 10000,
                                        allow_f_increases = true,
                                        outer_iterations = 10));
    
    
    
    # Bootstrap SE
    B = 100

    sol_mat_W_log_r = zeros(B, p)
    weight_draw_mat = zeros(B,N)
    f_w = Dirichlet(N,0.5)

    # for i in 1:B
    Threads.@threads for i in 1:B
            println("Iteration number ", i)
            w_draws = rand(f_w,1);
            weight_draw_mat[i,:] = w_draws;

            W_b = inv((w_draws.*data.Z_full)'*(w_draws.*data.Z_full)/N);
            fun_gmm_logs_b_W(x) = gmm_logs_bayesian(x,W_b,data,Pdims,w_draws, γ);
            @time results_gmm_loop_W = optimize(fun_gmm_logs_b_W,
                            lower,
                            upper,
                            x_init,
                            Fminbox(BFGS()),
                            autodiff = :forward,
                            Optim.Options(f_tol = tol,
                                        x_tol = tol,
                                        g_tol = tol,
                                        show_trace = false,
                                        show_every = 10000,
                                        iterations = 10000,
                                        allow_f_increases = true,
                                        outer_iterations = 30));

            sol_mat_W_log_r[i,:] = results_gmm_loop_W.minimizer;

            println("Iteration number ", i,": Nonparametric bootsrap in logs solved with ", results_gmm_loop_W.iterations, " iterations and minimum ", results_gmm_loop_W.minimum)

    end


    # Save matrix of results
    cd(main_path*"/output/estimates/")
    sol_mat_W_log_r = DataFrame(sol_mat_W_log_r,:auto)
    sol_mat_W_log_r = CSV.write("B_matrix_amenity_supply_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",sol_mat_W_log_r)

    # Read results
    cd(main_path*"/output/estimates/")
    sol_mat_W_log_r = DataFrame(CSV.File("B_matrix_amenity_supply_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv"))

    # mean
    sol_mat_W_log_r = Matrix{Float64}(sol_mat_W_log_r)
    mean_log_r = vec(mapslices(mean, sol_mat_W_log_r; dims=1))
    mean_log_mat_box_r = reshape(mean_log_r[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);

    # Compute residuals
    resid = log.(Y) - predict_logs(mean_log_r,data,Pdims, γ)
    SSR_robust = sum(resid.^2)
    resid_mat = zeros(N,4)
    resid_mat[:,1] = df.s
    resid_mat[:,2] = df.gb
    resid_mat[:,3] = df.year
    resid_mat[:,4] = resid
    resid_mat = DataFrame(resid_mat,:auto)

    # Save mean results
    mean_log_mat = zeros(length(mean_log_r),1)
    mean_log_mat[:,1] = mean_log_r
    mean_log_mat = DataFrame(mean_log_mat,:auto)
    CSV.write("B_amenity_supply_estimation_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",mean_log_mat)
    CSV.write("B_amenity_supply_residuals_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",resid_mat)

    # 10% significance
    top_percentile(x) = percentile(x, 95)
    bottom_percentile(x) = percentile(x, 5)
    ub_log = mapslices(top_percentile, sol_mat_W_log_r; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log_r; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_90_r = [lb_log; ub_log]
    CI_log_90_r = DataFrame(CI_log_90_r,:auto)
    CSV.write("B_CI90_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",CI_log_90_r)

    # 5% significance
    top_percentile(x) = percentile(x, 97.5)
    bottom_percentile(x) = percentile(x, 2.5)
    ub_log = mapslices(top_percentile, sol_mat_W_log_r; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log_r; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_95_r = [lb_log; ub_log]
    CI_log_95_r = DataFrame(CI_log_95_r,:auto)
    CSV.write("B_CI95_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",CI_log_95_r)

    # 1% significance
    top_percentile(x) = percentile(x, 99.5)
    bottom_percentile(x) = percentile(x, 0.5)
    ub_log = mapslices(top_percentile, sol_mat_W_log_r; dims=1)
    ub_log_mat_box = reshape(ub_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    lb_log = mapslices(bottom_percentile, sol_mat_W_log_r; dims=1)
    lb_log_mat_box = reshape(lb_log[1:Pdims.K*Pdims.S],Pdims.K,Pdims.S);
    CI_log_99_r = [lb_log; ub_log]
    # Save matrix of results
    CI_log_99_r = DataFrame(CI_log_99_r,:auto)
    CSV.write("B_CI99_gamma_"*string(round(-γ, digits = 2))*"_robustness.csv",CI_log_99_r)

    # Difference in means tests -> using CI
    CI_90_restricted = Matrix{Float64}(CI_log_90_r)[:,1:Pdims.K*Pdims.S]
    CF_90_main = Matrix{Float64}(CI_log_90)[:,1:Pdims.K*Pdims.S]
    overlap_90 = Vector{Float64}(CI_90_restricted[1,:].<CF_90_main[2,:]).*Vector{Float64}(CF_90_main[1,:].<CI_90_restricted[2,:])
    reshape(overlap_90,7,6)

    CI_95_restricted = Matrix{Float64}(CI_log_95_r)[:,1:Pdims.K*Pdims.S]
    CF_95_main = Matrix{Float64}(CI_log_95)[:,1:Pdims.K*Pdims.S]
    overlap_95 = Vector{Float64}(CI_95_restricted[1,:].<CF_95_main[2,:]).*Vector{Float64}(CF_95_main[1,:].<CI_95_restricted[2,:])
    reshape(overlap_95,7,6)

    CI_99_restricted = Matrix{Float64}(CI_log_99_r)[:,1:Pdims.K*Pdims.S]
    CF_99_main = Matrix{Float64}(CI_log_99)[:,1:Pdims.K*Pdims.S]
    overlap_99 = Vector{Float64}(CI_99_restricted[1,:].<CF_99_main[2,:]).*Vector{Float64}(CF_99_main[1,:].<CI_99_restricted[2,:])
    reshape(overlap_99,7,6)

    # Total number of parameters
    num_params = Pdims.K * Pdims.S  # Should be 7 * 6 = 42

    # Ensure that CI matrices have the correct size
    CI_log_90_r = CI_log_90_r[:, 1:num_params]
    CI_log_90_r = Matrix(CI_log_90_r)
    
    CI_log_95_r = CI_log_95_r[:, 1:num_params]
    CI_log_95_r = Matrix(CI_log_95_r)

    CI_log_99_r = CI_log_99_r[:, 1:num_params]
    CI_log_99_r = Matrix(CI_log_99_r)


    # Reshape the data
    CI90_lb_r = reshape(CI_log_90_r[1, :], Pdims.K, Pdims.S)

    CI95_lb_r = reshape(CI_log_95_r[1, :], Pdims.K, Pdims.S)

    CI99_lb_r = reshape(CI_log_99_r[1, :], Pdims.K, Pdims.S)

    epsilon = 1e-3
    stars_r = zeros(Int,Pdims.K,Pdims.S)
    # Determine the number of stars for each estimate
    for i in 1:Pdims.K
        for j in 1:Pdims.S
            if CI99_lb_r[i,j] > epsilon
                stars_r[i,j] = 3  # Significant at 1%
            elseif CI95_lb_r[i,j] > epsilon
                stars_r[i,j] = 2  # Significant at 5%
            elseif CI90_lb_r[i,j] > epsilon
                stars_r[i,j] = 1  # Significant at 10%
            else
                stars_r[i,j] = 0  # Not significant
            end
        end
    end

    display(stars_r)



    # Construct latex table
    df_output = Matrix{Float64}(CI_log_95_r[:,1:6*7])
    df_output_lb = vec(df_output[1,:])
    r_df_output_lb = reshape(df_output_lb,7,6)
    df_output_ub = vec(df_output[2,:])
    r_df_output_ub = reshape(df_output_ub,7,6)

    # Attach names
    group_names = ["Older Families";"Singles"; "Younger Families" ;"Students" ;"Immigrant Families" ;"Dutch Low Income" ;"Tourists"]
    r_df_output = hcat(group_names,mean_log_mat_box_r)
    amenity_names = ["Touristic Amenities" "Restaurants" "Bars" "Food Stores" "Non-Food Stores" "Nurseries"]
    amenity_string = ""
    for i in 1:6
    amenity_string *= "&"*amenity_names[i]
    end

    # Generate table header
    Table  = "\\begin{table}[!ht]\n";
    Table *= "\\setlength{\\tabcolsep}{0pt}\n";
    Table *= "\\centering \n";
    Table *= "\\caption{Estimates of amenity supply parameters.}\n";
    Table *= "\\label{tab:amenity_supply_estimation_robustness}\n";

    Table *= "\\scalebox{0.64}{\\begin{tabular}{lcccccc}\n";
    Table *= "    \\toprule\n";

    Table *=  amenity_string*"\\\\\n";
    Table *= "    \\midrule\n";
    # Generate table body
    for row in 1:7
        Table *= r_df_output[row,1]
        for column in 2:7
            estimate = round(r_df_output[row,column],digits=3)
            num_stars = stars_r[row, column-1]
            star_string = "*"^num_stars
            Table *= " & " * string(estimate) * star_string
        end
        Table *= "\\\\\n";
        for col in 1:6
            Table *= " & [" * string(round(r_df_output_lb[row,col],digits=3)) *","*string(round(r_df_output_ub[row,col],digits=3)) *"]"
        end
        Table *= "\\\\\n";
    end
    Table *= "  \\bottomrule\n";
    Table *= "\\end{tabular}}\n";
    Table *= """
    \\begin{minipage}{\\textwidth}{\\scriptsize Notes: Table reports bootstrap results for coefficients \$\\beta^k_s\$ from Equation \\ref{eq:amenities_regression_robustness} for seven population types and six types of services. Parameters \$\\beta^k_s\$ along with fixed effects \$\\lambda_j\$ and \$\\lambda_{p(j)t}\$ are estimated via GMM, where we restrict \$\\beta^k_s\\ge0\$. The estimation procedure is outlined in section \\ref{sec: amenity_supply estimation} and follows a Bayesian-bootstrap with random Dirichlet weights across 100 draws. Top rows indicate average estimates of the bootstrap samples. Results inside square brackets indicate 95\\% confidence intervals. We omit estimates of the location and time fixed effects. \\sym{*}\\(p<0.10\\), \\sym{**}\\(p<0.05\\), \\sym{***}\\(p<0.01\\).}
        
    \\end{minipage}
    """
    Table *= "\\end{table}\n";
    Table *= "\n";

    # Export result to .tex file
    cd(main_path*"/output/tables/")
    write("table_output_amenity_supply_robustness_gamma_"*string(round(-γ, digits = 2))*".tex", Table);
    
end

driver(-1/0.66, 10^(3), true) #Saiz for San Francisco 
#driver(-1/0.75, 10^(3), false) #Saiz for New York
#driver(-1/0.86, 10^(4), false) #Saiz for Boston
#driver(-1/1.07, 10^(4), false) #Saiz for Portland 
#driver(-1/1.24, 10^(4), false) #Saiz for Detroit
#driver(-1/1.61, 10^(6), false) #Saiz for Washington
#driver(-1/2.11, 10^(6), false) # Saiz Raleigh-Durham-Chapel Hill
#driver(-1/2.55, 10^(7), false) #Saiz for Atlanta