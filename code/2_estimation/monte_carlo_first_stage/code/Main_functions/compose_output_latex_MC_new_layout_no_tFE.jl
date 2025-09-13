####################################################################################  
## Code: MC LaTeX Tables
## Authors: Marek Bojko & Sriram Tolety 
####################################################################################  

using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Plots, Printf

cd("../../")
include("MC_functions_full_dynamic_corrected_stochastic_loc_cap.jl")

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Fix seed for reproducibility
Random.seed!(1243)


function reshape_output_df(df,n_specs)
  alpha_coeffs = reshape(Matrix(df[!,r"ab_alpha"])',3*nrow(df))
  beta_1_coeffs = reshape(Matrix(df[!,r"ab_beta_1"])',3*nrow(df))
  beta_2_coeffs = reshape(Matrix(df[!,r"ab_beta_2"])',3*nrow(df))
  gamma_0_coeffs = reshape(Matrix(df[!,r"ab_gamma_0"])',3*nrow(df))
  gamma_1_coeffs = reshape(Matrix(df[!,r"ab_gamma_1"])',3*nrow(df))
  gamma_2_coeffs = reshape(Matrix(df[!,r"ab_gamma_2"])',3*nrow(df))
  delta_coeffs = reshape(Matrix(df[!,r"ab_delta"])',3*nrow(df))
  pop_col = repeat(vec(df[!,"Pop"]),inner=3)
  J_col = repeat(vec(df[!,"J"]),inner=3)
  xi_col = repeat(vec(df[!,"xi"]),inner=3)
  prob_col = repeat(["T","L","F"],outer = n_specs)

  df_output = DataFrame(xi = xi_col, Pop = pop_col, J = J_col, Prob = prob_col, alpha = alpha_coeffs,
                          beta_1 = beta_1_coeffs, beta_2 = beta_2_coeffs, gamma_0 = gamma_0_coeffs,
                          gamma_1 = gamma_1_coeffs, gamma_2 = gamma_2_coeffs, delta = delta_coeffs)
  return df_output
end


# choose layout (vertical or horizontal)
layout = "vertical"

#Dimensions
S = 2
TT = 10
tau_bar = 2
n_param = 5

# Compute avg bias for different parameter specifications
J_to_consider = [25]
Pops_to_consider = [50000,1000050]
FEs_to_consider = [true]
tau_to_consider = [true]
divide_by_params_options = [false,true]

# True parameters
α = -0.05
β = 0.1*ones(Float64,S)
#γ = -0.0025
gamma_0 = -0.0025
gamma_1 = -0.0025
gamma_2 = -0.5
δ = 0.1
true_param = [α;β;gamma_0;gamma_1;gamma_2;δ]
n_param = length(true_param)

n_specs = length(J_to_consider)*length(Pops_to_consider)*length(FEs_to_consider)*length(tau_to_consider)*3


for include_tau_dummy in tau_to_consider
   for FEs in FEs_to_consider
      for divide_by_params in divide_by_params_options

         # Compute the number of inner iterations (i.e. how many times do we draw a dataset?)
         n_iter = 10

         #Dimensions
         S = 2
         TT = 10
         tau_bar = 2
         n_param = length(true_param)

         Pdims = (S = S,
                  T = TT,
                  FEs = FEs,
                  J = 25,
                  include_tau_dummy = include_tau_dummy,
                  tau_bar = tau_bar,
                  divide_by_params = divide_by_params)

         #### Compare output and output to latex
         df_output_zero = DataFrame(CSV.File("Output/avg_bias_experiments_MNL_full_dynamic_25locs_xizero_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_cr.csv"))
         df_output_zero[!,"xi"] = ["zero" for i in 1:length(Pops_to_consider)]

         df_output_exo = DataFrame(CSV.File("Output/avg_bias_experiments_MNL_full_dynamic_25locs_xiexogenous_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_cr.csv"))
         df_output_exo[!,"xi"] = ["exogenous" for i in 1:length(Pops_to_consider)]
         df_output_exo[!,Not([:Pop,:J,:xi])] = 10/9 .* df_output_exo[!,Not([:Pop,:J,:xi])]

         df_output_endo = DataFrame(CSV.File("Output/avg_bias_experiments_MNL_full_dynamic_25locs_xiendogenous_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_cr.csv"))
         df_output_endo[!,"xi"] = ["endogenous" for i in 1:length(Pops_to_consider)]
         df_output_endo[!,Not([:Pop,:J,:xi])] = 10/8 .* df_output_endo[!,Not([:Pop,:J,:xi])]

         # Append
         df_all = df_output_zero
         append!(df_all,df_output_exo)
         append!(df_all,df_output_endo)

         # Save
         CSV.write("Output/avg_bias_experiments_MNL_full_dynamic_25locs_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE.csv",  df_all)

         # Perform some normalization
         df_all = df_all[!,Not(Between(:b_alpha_true,:b_alpha_freq))]
         df_all = df_all[!,Not(Between(:b_beta_1_true,:b_beta_1_freq))]
         df_all = df_all[!,Not(Between(:b_beta_2_true,:b_beta_2_freq))]
         df_all = df_all[!,Not(Between(:b_gamma_0_true,:b_gamma_0_freq))]
         df_all = df_all[!,Not(Between(:b_gamma_1_true,:b_gamma_1_freq))]
         df_all = df_all[!,Not(Between(:b_gamma_2_true,:b_gamma_2_freq))]
         df_all = df_all[!,Not(Between(:b_delta_true,:b_delta_freq))]

         if Pdims.divide_by_params
            df_all[!,r"alpha"] = abs.(df_all[!,r"alpha"]./α)
            df_all[!,r"beta_1"] = abs.(df_all[!,r"beta_1"]./β[1])
            df_all[!,r"beta_2"] = abs.(df_all[!,r"beta_2"]./β[2])
            df_all[!,r"gamma_0"] = abs.(df_all[!,r"gamma_0"]./gamma_0)
            df_all[!,r"gamma_1"] = abs.(df_all[!,r"gamma_1"]./gamma_1)
            df_all[!,r"gamma_2"] = abs.(df_all[!,r"gamma_2"]./gamma_2)
            df_all[!,r"delta"] = abs.(df_all[!,r"delta"]./δ)
         end

         # transform to vertical
         df_horizontal = reshape_output_df(df_all,n_specs)

         # Save
         CSV.write("Output/avg_bias_experiments_MNL_full_dynamic_25locs_FE$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_no_tFE_vertical.csv",  df_horizontal)


         #######################################################################
         # Generate latex table
         #######################################################################

         if layout == "horizontal"
           # Generate table header
           ### table1.jl script
           # Generate table header
           Table  = "\\begin{table}[H]\n";
           Table *= "\\resizebox{\\columnwidth}{!}{%\n";
           Table *= "\\begin{threeparttable}\n";
           if Pdims.FEs == true
             if Pdims.divide_by_params
               if Pdims.include_tau_dummy
                 Table *= "\\caption{Monte Carlo simulations with fixed effects and an indicator for high location capital, normalized}\n";
                 Table *= "\\label{tab: MCs FEs loc dummy normalized}\n";
               else
                 Table *= "\\caption{Monte Carlo simulations with fixed effects and location capital, normalized}\n";
                 Table *= "\\label{tab: MCs FEs loc normalized}\n";
               end
             else
               if Pdims.include_tau_dummy
                 Table *= "\\caption{Monte Carlo simulations with fixed effects and an indicator for high location capital, absolute values}\n";
                 Table *= "\\label{tab: MCs FEs loc dummy not normalized}\n";
               else
                 Table *= "\\caption{Monte Carlo simulations with fixed effects and location capital, absolute values}\n";
                 Table *= "\\label{tab: MCs FEs loc not normalized}\n";
               end
             end
           else
             if Pdims.divide_by_params
               Table *= "\\caption{Monte Carlo simulations without fixed effects, normalized}\n";
               Table *= "\\label{tab: MCs no FEs normalized}\n";
             else
               Table *= "\\caption{Monte Carlo simulations without fixed effects, absolute values}\n";
               Table *= "\\label{tab: MCs no FEs not normalized}\n";
             end
           end
           Table *= "\\begin{tabular}{ll @{\\hspace{1cm}}"* ("cccc")^(n_param) * "}\n";
           Table *= "    \\hline\n";
           Table *= "    % Table header\n";
           Table *= "    & &\\multicolumn{3}{c}{\$ \\alpha \$} & & \\multicolumn{3}{c}{\$ \\beta_1 \$} & & \\multicolumn{3}{c}{\$ \\beta_2 \$}  & & \\multicolumn{3}{c}{\$ \\gamma_0 \$} & & \\multicolumn{3}{c}{\$ \\gamma_1 \$} & & \\multicolumn{3}{c}{\$ \\gamma_2 \$} & & \\multicolumn{3}{c}{\$ \\delta \$} \\\\ \\cline{3-5} \\cline{7-9} \\cline{11-13} \\cline{15-17} \\cline{19-21} \\cline{23-25} \\cline{27-29} \n";
           Table *= "    \$ \\xi \$ & Pop (in \$10^3\$) & T & L & F & & T & L & F & & T & L & F  & & T & L & F & & T & L & F & & T & L & F  & & T & L & F \\\\\n";
           Table *= "    \\hline\n";
           Table *= "    \\hline\n";
           # Generate table body (with nice alternating row colours)
           Table *= "    % Table body\n";
           for row in 1:nrow(df_all)
             if (row-1)%2 == 0
               Table *= "  " * df_all[row,"xi"] * " & " * string(Int(round(df_all[row,"Pop"]/1000,digits=0)));
               for col in 2:(ncol(df_all)-1)
                 if col%3 != 1
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]);
                 else
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]) * " & ";
                 end
               end
               Table *= " \\\\\n";
             elseif (row)%2 == 0
               Table *= "   & " * string(Int(round(df_all[row,"Pop"]/1000,digits=0)));
               for col in 2:(ncol(df_all)-1)
                 if col%3 != 1
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]);
                 else
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]) * " & ";
                 end
               end
               Table *= " \\\\\n";
               if row != nrow(df_all)
                 Table *= (" & ")^(4*n_param+1) * " \\\\ \n";
               end
             else
               Table *= "   & " * string(Int(round(df_all[row,"Pop"]/1000,digits=0)));
               for col in 2:(ncol(df_all)-1)
                 if col%3 != 1
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]);
                 else
                   Table *= " & " * @sprintf("%.1E",df_all[row,col]) * " & ";
                 end
               end
               Table *= " \\\\\n";
             end
           end
           Table *= "  \\hline\n";
           Table *= "  \\hline\n";
           Table *= "\\end{tabular}\n";
           Table *= "\\begin{tablenotes}\n"
           Table *= "\\end{tablenotes}\n"
           Table *= "\\small\n"
           if Pdims.divide_by_params
             Table *= "\\textit{Notes:} The table presents averaged absolute distance between the estimated parameter and the true parameter, divided by the true parameter, over 10 random draws of datasets. T represents estimation using the true transition probabilities; L represents estimation using predicted probabilities by a multi-nomial logit model; and F represents estimation using transition probabilities computed based on empirical shares.\n"
           else
             Table *= "\\textit{Notes:} The table presents averaged absolute distance between the estimated parameter and the true parameter, over 10 random draws of datasets. T represents estimation using the true transition probabilities; L represents estimation using predicted probabilities by a multi-nomial logit model; and F represents estimation using transition probabilities computed based on empirical shares.\n"
           end
           Table *= "\\end{threeparttable}}\n"
           Table *= "\\end{table}\n";
        else
          # Generate table header
          ### table1.jl script
          # Generate table header
          Table  = "\\begin{table}[h!]\n";
          Table *= "\\resizebox{\\columnwidth}{!}{%\n";
          Table *= "\\begin{threeparttable}\n";
          if Pdims.FEs == true
            if Pdims.divide_by_params
              if Pdims.include_tau_dummy
                Table *= "\\caption{Monte Carlo simulations with location fixed effects only and an indicator for high location capital, normalized}\n";
                Table *= "\\label{tab: MCs FEs loc dummy normalized}\n";
              else
                Table *= "\\caption{Monte Carlo simulations with location fixed effects only and location capital, normalized}\n";
                Table *= "\\label{tab: MCs FEs loc normalized}\n";
              end
            else
              if Pdims.include_tau_dummy
                Table *= "\\caption{Monte Carlo simulations with location fixed effects only and an indicator for high location capital, absolute values}\n";
                Table *= "\\label{tab: MCs FEs loc dummy not normalized}\n";
              else
                Table *= "\\caption{Monte Carlo simulations with location fixed effects only and location capital, absolute values}\n";
                Table *= "\\label{tab: MCs FEs loc not normalized}\n";
              end
            end
          else
            if Pdims.divide_by_params
              Table *= "\\caption{Monte Carlo simulations without fixed effects, normalized}\n";
              Table *= "\\label{tab: MCs no FEs normalized}\n";
            else
              Table *= "\\caption{Monte Carlo simulations without fixed effects, absolute values}\n";
              Table *= "\\label{tab: MCs no FEs not normalized}\n";
            end
          end
          Table *= "\\begin{tabular}{lll @{\\hspace{1cm}}"* ("cc")^(n_param) * "}\n";
          Table *= "    \\hline\n";
          Table *= "    % Table header\n";
          Table *= "    & & & \\multicolumn{13}{c}{ Mean of the absolute value of bias } \\\\ \\cline{4-16} \n"
          Table *= "    \$ \\xi \$ & Pop (in \$10^3\$) & Prob. & \$ \\alpha \$  & & \$ \\beta_1 \$ & & \$ \\beta_2 \$ & & \$ \\gamma_0 \$ & & \$ \\gamma_1 \$ & & \$ \\gamma_2 \$ & & \$ \\delta \$ \\\\ \n";
          Table *= "    \\hline\n";
          Table *= "    \\hline\n";
          # Generate table body (with nice alternating row colours)
          Table *= "    % Table body\n";
          for row in 1:nrow(df_horizontal)
            if (((row-1)%6 == 0) && (row%3 != 0))
              Table *= "  " * df_horizontal[row,"xi"] * " & " * string(Int(round(df_horizontal[row,"Pop"]/1000,digits=0))) * " & " * df_horizontal[row,"Prob"];
              for col in 5:ncol(df_horizontal)-1
                Table *= " & " * @sprintf("%.1E",df_horizontal[row,col]) * " & ";
              end
              Table *= " & " * @sprintf("%.1E",df_horizontal[row,ncol(df_horizontal)]);
              Table *= " \\\\\n";
            elseif ((row-1)%6 != 0 && (row-1)%3 == 0)
              Table *= "   & " * string(Int(round(df_horizontal[row,"Pop"]/1000,digits=0))) * " & " * df_horizontal[row,"Prob"];
              for col in 5:ncol(df_horizontal)-1
                Table *= " & " * @sprintf("%.1E",df_horizontal[row,col]) * " & ";
              end
              Table *= " & " * @sprintf("%.1E",df_horizontal[row,ncol(df_horizontal)]);
              Table *= " \\\\\n";
            elseif row%6 == 0
              Table *= "   &  & " * df_horizontal[row,"Prob"];
              for col in 5:ncol(df_horizontal)-1
                Table *= " & " * @sprintf("%.1E",df_horizontal[row,col]) * " & ";
              end
              Table *= " & " * @sprintf("%.1E",df_horizontal[row,ncol(df_horizontal)]);
              Table *= " \\\\\n";
              if row != nrow(df_horizontal)
                Table *= "\\hline \n";
              end
            else
              Table *= "   &  & " * df_horizontal[row,"Prob"];
              for col in 5:ncol(df_horizontal)-1
                Table *= " & " * @sprintf("%.1E",df_horizontal[row,col]) * " & ";
              end
              Table *= " & " * @sprintf("%.1E",df_horizontal[row,ncol(df_horizontal)]);
              Table *= " \\\\\n";
            end
          end
          Table *= "  \\hline\n";
          Table *= "  \\hline\n";
          Table *= "\\end{tabular}\n";
          Table *= "\\begin{tablenotes}\n"
          Table *= "\\item[]\n"
          Table *= "\\end{tablenotes}\n"
          Table *= "\\small\n"
          if Pdims.divide_by_params
            Table *= "\\textit{Notes:} The table presents averaged absolute distance between the estimated parameter and the true parameter, divided by the true parameter, over 10 random draws of datasets. T represents estimation using the true transition probabilities; L represents estimation using predicted probabilities by a multi-nomial logit model; and F represents estimation using transition probabilities computed based on empirical shares.\n"
          else
            Table *= "\\textit{Notes:} The table presents averaged absolute distance between the estimated parameter and the true parameter, over 10 random draws of datasets. T represents estimation using the true transition probabilities; L represents estimation using predicted probabilities by a multi-nomial logit model; and F represents estimation using transition probabilities computed based on empirical shares.\n"
          end
          Table *= "\\end{threeparttable}}\n"
          Table *= "\\end{table}\n";
        end

         # Export result to .tex file
         if layout == "horizontal"
           write("Output/table_FEs$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_div$(Pdims.divide_by_params)_no_tFE_horizontal.tex", Table);
         else
           write("Output/table_FEs$(Pdims.FEs)_locdummy$(Pdims.include_tau_dummy)_div$(Pdims.divide_by_params)_no_tFE_vertical.tex", Table);
        end
      end
   end
end
