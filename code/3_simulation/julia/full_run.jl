####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

################################################################################
# Set-up
################################################################################

# Packages
using Random, LinearAlgebra, DataFrames, CSV, Distributions, StatsBase, GLM, Plots, Printf

using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Setting spec 
gamma = -1/0.66
gamma_files = "152"
B_option = true
year = 2017
myopic = false
static = false

# File options
B_file = ifelse(B_option,"B_","")
myopic_file = ifelse(static,"_static","")
static_file = ifelse(static,"_static","")
beta = ifelse(myopic,0,0.85)
include_mc = ifelse(static,0,1)

# Functions
include("model_endogenous_tourists.jl")
include("sanity_checks.jl")
include("baseline_eq_no_airbnb.jl")
include("baseline_eq.jl")
include("baseline_eq_homogeneous.jl")
include("CF_decomposition_baseline.jl")
include("CF_tax.jl")
include("stability_endo_eq.jl")