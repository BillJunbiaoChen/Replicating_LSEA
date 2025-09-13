###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Prepare choice probability files
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

import Pkg
Pkg.precompile()
Pkg.add("Random")
Pkg.add("Distributions")
Pkg.add("LinearAlgebra")
Pkg.add("Optim")
Pkg.add("ForwardDiff")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
Pkg.add("Combinatorics")
Pkg.add("GLM")
Pkg.add("XLSX")

using Random, Distributions, LinearAlgebra, Optim, ForwardDiff, CSV, DataFrames, StatsBase, Combinatorics, GLM, XLSX, UnPack

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))


################################################################################
# Set-up
################################################################################

# Set path
path = @__DIR__
cd(path)

pwd()


println(pwd())
include("estimation_functions_reduced.jl")

#Dimensions
S = 5
TT = 12
tau_bar = 2
J = 23
K = 3
lower_bound_year_sample = 2008

global Pdims = (S = S,
         J = J,
         K = K,
         T = TT,
         FEs = true,
         tau_bar = tau_bar,
         lower_bound_year_sample = lower_bound_year_sample)

# Discount rate
rho = 0.85

Pparam = (ρ = rho,
            fill_prob = 10^-6)

####################################################################################
# Prepare parameters and data
####################################################################################

# Read distance matrix
cd("../../data")
dist_mat = Matrix{Float64}(DataFrame(CSV.File("final/inputs/dist_mat_centroids.csv")))

#### Iterate over groups

for g in 1:Pdims.K

    println("----------------------------------------------------------")
    println("----------------------------------------------------------")
    println("Group: $g")
    println("----------------------------------------------------------")
    println("----------------------------------------------------------")

    # Prepare datasets
    MNL_probs = DataFrame(CSV.File("cbs/exports/Vrijgegeven220328_0500_MNL Coefficients - April 2022/predicted_probs_MNL_$g.csv"))
    tau_trans_probs = DataFrame(CSV.File("cbs/exports/220331_0500_Transition location tenure - April 2022/tau_transition_probs_$g.csv"))

    #### Prepare matrix of MNL probabilities
    # Rename option 0 to J
    MNL_probs[MNL_probs.gb .== 0,:gb] .= J
    MNL_probs[MNL_probs.p_gb .== 0,:p_gb] .= J

    # Keep years Pdims.lower_bound_year_sample - 2019
    MNL_probs = MNL_probs[(MNL_probs.year .>= lower_bound_year_sample) .&  (MNL_probs.year .<= 2019),:]

    # Define time period
    MNL_probs[!,"t"] .= 0
    for t in 1:Pdims.T
        MNL_probs[MNL_probs.year .== 2007+t,:t] .= t
    end

    # Reshape to wide
    sort!(MNL_probs, [:gb,:p_gb,:p_tau,:year])
    MNL_probs_unstacked = unstack(MNL_probs,[:p_gb,:p_tau,:year,:t], :gb, :p_hat, renamecols=x->Symbol(:phat, x),allowduplicates=true)
    sort!(MNL_probs_unstacked, [:p_gb,:p_tau,:year])

    #### Prepare empirical transitions for location capital
    # Rename option 0 to J
    tau_trans_probs[tau_trans_probs.gb .== 0,:gb] .= J

    # Keep years Pdims.lower_bound_year_sample - 2019
    tau_trans_probs = tau_trans_probs[(tau_trans_probs.year .>= lower_bound_year_sample) .&  (tau_trans_probs.year .<= 2019),:]

    # Define time period
    tau_trans_probs[!,"t"] .= 0
    for t in 1:Pdims.T
        tau_trans_probs[tau_trans_probs.year .== 2007+t,:t] .= t
    end

    # Define the Gamma tensor for stochastic evolution of the location capital
    Gamma_tensor = Γ_tensor(Pdims,tau_trans_probs)
    Pgt = (Gamma_tensor = Gamma_tensor,)
    global Pdims = merge(Pdims,Pgt)


    ########################################################################
    # Prepare ECCP paths and invariant data
    ########################################################################

    df_all = prepare_invariant_data(MNL_probs_unstacked,dist_mat,Pdims,Pparam,g)

    # Export
    CSV.write("constructed/prep_reduced_eccp_data_g_$(g).csv",  df_all)

end



# Discount rate
rho = 0.0

Pparam = (ρ = rho,
            fill_prob = 10^-6)

####################################################################################
# Prepare parameters and data
####################################################################################

# Read distance matrix
dist_mat = Matrix{Float64}(DataFrame(CSV.File("final/inputs/dist_mat_centroids.csv")))

#### Iterate over groups

for g in 1:Pdims.K

    println("----------------------------------------------------------")
    println("----------------------------------------------------------")
    println("Group: $g")
    println("----------------------------------------------------------")
    println("----------------------------------------------------------")

    # Prepare datasets
    MNL_probs = DataFrame(CSV.File("cbs/exports/Vrijgegeven220328_0500_MNL Coefficients - April 2022/predicted_probs_MNL_$g.csv"))
    tau_trans_probs = DataFrame(CSV.File("cbs/exports/220331_0500_Transition location tenure - April 2022/tau_transition_probs_$g.csv"))

    #### Prepare matrix of MNL probabilities
    # Rename option 0 to J
    MNL_probs[MNL_probs.gb .== 0,:gb] .= J
    MNL_probs[MNL_probs.p_gb .== 0,:p_gb] .= J

    # Keep years Pdims.lower_bound_year_sample - 2019
    MNL_probs = MNL_probs[(MNL_probs.year .>= lower_bound_year_sample) .&  (MNL_probs.year .<= 2019),:]

    # Define time period
    MNL_probs[!,"t"] .= 0
    for t in 1:Pdims.T
        MNL_probs[MNL_probs.year .== 2007+t,:t] .= t
    end

    # Reshape to wide
    sort!(MNL_probs, [:gb,:p_gb,:p_tau,:year])
    MNL_probs_unstacked = unstack(MNL_probs,[:p_gb,:p_tau,:year,:t], :gb, :p_hat, renamecols=x->Symbol(:phat, x),allowduplicates=true)
    sort!(MNL_probs_unstacked, [:p_gb,:p_tau,:year])

    #### Prepare empirical transitions for location capital
    # Rename option 0 to J
    tau_trans_probs[tau_trans_probs.gb .== 0,:gb] .= J

    # Keep years Pdims.lower_bound_year_sample - 2019
    tau_trans_probs = tau_trans_probs[(tau_trans_probs.year .>= lower_bound_year_sample) .&  (tau_trans_probs.year .<= 2019),:]

    # Define time period
    tau_trans_probs[!,"t"] .= 0
    for t in 1:Pdims.T
        tau_trans_probs[tau_trans_probs.year .== 2007+t,:t] .= t
    end

    # Define the Gamma tensor for stochastic evolution of the location capital
    Gamma_tensor = Γ_tensor(Pdims,tau_trans_probs)
    Pgt = (Gamma_tensor = Gamma_tensor,)
    global Pdims = merge(Pdims,Pgt)


    ########################################################################
    # Prepare ECCP paths and invariant data
    ########################################################################

    df_all = prepare_invariant_data(MNL_probs_unstacked,dist_mat,Pdims,Pparam,g)

    # Export
    CSV.write("constructed/prep_reduced_eccp_data_g_$(g)_static.csv",  df_all) #setting rho = 0

end