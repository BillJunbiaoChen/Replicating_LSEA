####################################################################################  
## Code: Model Estimation
## Author: Milena almagro
####################################################################################  

###################################################################################################
## 1. AUXILIARY OBJECTS
###################################################################################################

"""
function dist_est(P::NamedTuple,k::Int64)
    Computes the matrix of moving costs for type k.
"""
function dist_est(P::NamedTuple,k::Int64)
    ones_off_diag = ones(P.J,P.J) - diagm(ones(P.J))
    inner_mat = P.gamma_0[k] * ones_off_diag  + P.gamma_1[k] * P.dist_mat
    full_mat_h = hcat(inner_mat, P.gamma_2[k] * ones(P.J))
    full_mat = vcat(full_mat_h, P.gamma_2[k] *[ones(P.J);0]')
    return full_mat
end


"""
function MC_mat(P::NamedTuple,k::Int64)
    transforms a tensor of moving costs of agent types to a matrix indexed by [(j,τ),d,k]
"""
function MC_mat(P::NamedTuple)
    MCmat = zeros(2*(P.J+1),P.J+1,P.K)
    for k = 1:P.K
        MCmat[:,:,k] = kron(dist_est(P,k),ones(2))
    end
    return MCmat
end


"""
function T_stochastic_k(P::NamedTuple,tau_trans_probs::DataFrame,k::Int64)
    Constructs a tensor describing the evolution of location tenure for a give type. Rows are indexed by (j,τ), columns are indexed by (j',τ').
    Note about inputs: tau_trans_probs - DataFrame, a dataframe with estimated transition probabilities for location capital
"""
function T_stochastic(P::NamedTuple,tau_trans_probs::DataFrame)
    T_i_tensor = zeros(Float64,P.tau_bar*(P.J+1),P.tau_bar*(P.J+1),P.K)
    for k = 1: P.K
        mat = hcat(ones(P.tau_bar), zeros(P.tau_bar))
        T_i_tensor[:,:,k] = kron(ones(23,23),mat)
        for j in 1:(P.J + 1)
            trans_prob_12 = tau_trans_probs[(tau_trans_probs.gb .== j) .& (tau_trans_probs.combined_cluster .== k),"transition_prob"][1]
            mat = [(1-trans_prob_12) trans_prob_12; 0 1]
            T_i_tensor[(j-1)*P.tau_bar+1:j*P.tau_bar,(j-1)*P.tau_bar+1:j*P.tau_bar,k] = [(1-trans_prob_12) trans_prob_12; 0 1]

        end
    end
    return T_i_tensor
end

"""
function E_EV_k(EV::Vector{Float64},P::NamedTuple,k::Int64)

Computes the expected value function for a group k 
"""

function E_EV_k(EV::Vector{Float64},P::NamedTuple,k::Int64)
    T_k = P.T_stochastic[:,:,k]
    diag_EV = zeros((P.J+1)*P.tau_bar,(P.J+1))
    for j = 1:(P.J+1)
        diag_EV[(j-1)*P.tau_bar+1:j*P.tau_bar,j] = EV[(j-1)*P.tau_bar+1:j*P.tau_bar]
    end

    return T_k*diag_EV

end

"""
function stationary_dist_MC(M::AbstractMatrix{Float64})

Computes the stationary distribution of a Markov Chain given its transition matrix
"""
function stationary_dist_MC_iter(M::AbstractMatrix{Float64},P)

    initial_pi = ones((P.J+1)*P.tau_bar)./((P.J+1)*P.tau_bar)

    iter = 0
    tol = 1

    while (iter < 10^5) && (tol > 1e-8)
        new_pi = M'*initial_pi
        tol = norm(initial_pi-new_pi,Inf)
        initial_pi = copy(new_pi)
        iter = iter + 1
    end

    # Normalize and return
    return initial_pi ./ sum(initial_pi)
end

###################################################################################################
## 2. HOUSING DEMAND AND SUPPLY; AMENITIES
###################################################################################################

"""
function S_L(r::Vector{Float64},p::Vector{Float64},P::NamedTuple)
    returns the fraction of long-term houses for each location
"""
function S_L(r::Vector{Float64}, p::Vector{Float64}, P::NamedTuple,options)
    # We need to adjust the price of long-term rental, which is expressed in meters
    # To do so, we use the average squared footage in locations

    if P.include_airbnb == true 
        r_annual = r .* P.avg_squared_footage

        # Airbnb tax rate
        p_annual = (1 - options.airbnb_tax_rate)*p*365

        # Compute price gap
        price_gap = (r_annual-p_annual)./10^4

        SL = 1 ./(exp.(-P.alpha.*price_gap - P.kappa_j).+ 1)
    else
        SL = ones(P.J)
    end

    return SL
end

"""
function tourist_guest_demand(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    Computes discrete choice demand from tourists for Airbnb listings in different neighborhoods.
"""

# We need to change this function once we moved to new estimates 
function tourist_guest_demand(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)::Vector{Float64}
    p_guest = p./P.mean_accommodates
    y_hat_tourists = [[log.(p_guest);0] [log.(a); zeros(6)'] P.tourist_demand_controls]*P.θ_tourist
    δ_tourists = P.δ_j_tourist + y_hat_tourists
    δ_norm = δ_tourists[end-1]
    δ_tourists_norm = δ_tourists .- δ_norm
    E_tourists = exp.(δ_tourists_norm)
    prob_tourists = E_tourists./sum(E_tourists)
    guest_demand = prob_tourists*P.total_tourists
    return guest_demand
end

"""
function D_S(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    Computes demand from tourists for Airbnb listings in different neighborhoods.
"""
function D_S(r::Vector{Float64}, p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    # Return the total demand
    if P.include_airbnb
        if P.endogenous_tourist_choices 
            DS = tourist_guest_demand(p,a,P)[1:P.J] .* P.str_guests_to_total_guests 
        else
            DS = (1 .- S_L(r,p,P,options)) .* P.listings_to_total_str_guests .* P.H
        end
    else
        DS = zeros(P.J)
    end

    return DS
end 

"""
function hotel_pop(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    Computes total hotel population in different neighborhoods.
"""
function hotel_pop(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    if P.endogenous_tourist_choices 
        if P.include_airbnb
            tourists_hotels = tourist_guest_demand(p,a,P)[end]
            occupancy_rate = tourists_hotels/sum(P.hotel_beds)
            DH = occupancy_rate*P.hotel_beds
        else
            occupancy_rate = P.total_tourists/sum(P.hotel_beds)
            DH = occupancy_rate*P.hotel_beds 
        end
    else
        DH = P.pop_hotel_tourists 
    end
    return DH
end 



"""
function tourists_pop(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    Computes total hotel population in different neighborhoods.
"""
function tourists_pop(r::Vector{Float64},p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)

    tourist_demand = tourist_guest_demand(p,a,P)
    pop_hotels = zeros(P.J)
    pop_airbnb = zeros(P.J)

    # Construct hotel population 
    if P.endogenous_tourist_choices 
        if P.include_airbnb
            total_tourists_hotels = tourist_demand[end]
            occupancy_rate = total_tourists_hotels/sum(P.hotel_beds)
            pop_hotels = occupancy_rate*P.hotel_beds
            pop_airbnb = tourist_demand[1:P.J] .* P.str_guests_to_total_guests 
        else
            occupancy_rate = P.total_tourists/sum(P.hotel_beds)
            pop_hotels = occupancy_rate*P.hotel_beds 
            pop_airbnb = zeros(P.J)
        end
    else
        pop_hotels = P.pop_hotel_tourists 
        pop_airbnb = P.include_airbnb*(1 .- S_L(r,p,P,options)) .* P.listings_to_total_str_guests .* P.H
    end

    return pop_hotels, pop_airbnb
end 


"""
function D_sq_area(r::Vector{Float64},P::NamedTuple)
    Computes the demand for living area, conditional on living in the given location, returning a J x K matrix
"""
function D_sq_area(r::Vector{Float64},P::NamedTuple)
    CB_opt = kron(ones(P.J),((1 .- P.exp_shares[1:P.K]) .* P.income[1:P.K])') ./ r  # J by K matrix
    return CB_opt
end


"""
function Amenity_supply(r::Vector{Float64},p::Vector{Float64},a::Matrix{Float64},D::Matrix{Float64},P::NamedTuple)
    Computes supply of amenities in equilibrium, for a given vector of rental prices and demand in the steady state, for a model with homothetic preferences.
"""
function Amenity_supply(r::Vector{Float64},p::Vector{Float64},a::Matrix{Float64},D::Matrix{Float64},P::NamedTuple)
    
    #### Compute expenditure shares
    # Compue total amenities
    a_j = sum(a, dims = 2)

    # Compute the budget term
    budget = kron(ones(P.J),P.income') .* kron(ones(P.J),P.exp_shares')
    
    # Compute the total population of tourists
    pop_hotels, pop_airbnb = tourists_pop(r,p,a,P)
    total_pop_tourists = pop_hotels + pop_airbnb

    # Population counts
    pop_mat = [D P.pop_exogeneous total_pop_tourists]
    
    # Compute the total expenditure shares
    exp_share = budget.*pop_mat*P.alpha_ks

    # Compute the amenities supply equation from estimation (in logs)
    Log_Amenity_supply_endo = log.(exp_share ./ P.norm_amenities) .+ P.amenity_time_FE .+ P.amenity_loc_FEs .+ P.gamma*log.(a_j) .+ P.amenity_resid .- log(1+options.amenity_tax_rate)
    
    # Transform to levele
    Amenity_supply_endo = exp.(Log_Amenity_supply_endo)
    
    return Amenity_supply_endo
end

####### New utility function
function utility(r::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    θ_resident = P.θ_resident
    log_r_meter = log.(r)
    X_resident_exo_net_amenity = P.X_resident_exo
    X_resident = [log_r_meter log.(a) X_resident_exo_net_amenity  ]
    u = X_resident*θ_resident
    u = [u ;zeros(P.K)']
    return u
end




"""
function trans_mat_EV_mat_exo(r::Vector{Float64},a::Matrix{Float64},k::Int64,P::NamedTuple, EV_initial::Vector{Float64}=rand(P.tau_bar*(P.J+1)))
    Computes the EV function and a transition matrix for specified rental prices, amenities, and a household type. Allows the user to supply
        custom initial guess for the EV vector. Note that we always index the rows of objects by (j,τ) (e.g. the flow utility matrix is
        indexed by [(j,τ),d]).

    Output:
        - Pi_mat - a τ_bar x J square matrix giving the transition probabilities, states are indexed by (j,τ)
        - EV_G - a τ_bar x J vector giving the values of EV function. Entries are indexed by (j,τ)
"""
function trans_mat_EV(r::Vector{Float64},a::Matrix{Float64},k::Int64,P::NamedTuple,EV_initial::Vector{Float64}=rand(P.tau_bar*(P.J+1)))
    #### Compute flow utility
    u_hat_mat = utility(r,a,P)
    flow_U = kron(u_hat_mat[:,k]',ones(P.tau_bar*(P.J+1))) + #utility flow
             P.delta_tau[k] * P.T_stochastic[:,:,k]*kron(Matrix{Float64}(I,P.J+1,P.J+1),[ 0; 1] ) + # Location capital
             P.MC_mat[:,:,k]
    
    #### Compute the EV function
    # initial guess
    EV_G = EV_initial
    
    # Initialize variables
    dist_func = 1
    iter = 0

    # iterate
    while dist_func > options.tol_EV && iter <= options.max_iter_EV
        # Compute the expected continuation value by appropriately rearranging entries in EV
        DEVc_mat = E_EV_k(EV_G,P,k)

        # Prepare the inclusive value
        v = flow_U + P.beta*DEVc_mat

        # Overfloating adjustment
        v_max = maximum(v)
        v = v .- v_max

        # Update guess
        EV = log.(exp.(v)*ones(P.D)) .+ v_max
        
        # Compute convergence criterion
        dist_func = norm(EV-EV_G,Inf)

        # Update vars
        EV_G = copy(EV)
        iter += 1
    end

    #### Compute the transition matrix
    
    # Prepare a matrix of choice probabilities
    DEVc_mat = E_EV_k(EV_G,P,k)
    v = flow_U + P.beta*DEVc_mat
    v_max = maximum(v)
    v = v .- v_max
    denom = exp.(v)*ones(P.D)
    Q_mat = (exp.(v)) ./ denom

    # Get the transition matrix for location capital
    T_mat = P.T_stochastic[:,:,k]

    # Compute an auxiliary transition matrix
    Pi_mat = T_mat .* kron(Q_mat,ones(1,2))
    
    # Return the EV function and the transition matrix
    return Pi_mat, EV_G
end




"""
function D_L(r::Vector{Float64},a::Matrix{Float64},P::NamedTuple,EVs_initial::Matrix{Float64}=rand(P.tau_bar*(P.J+1),P.K))
    For a vector of rental prices and amenities, computes the demand for long-term housing based on the stationary distribution

    Output:
    - D_Ls - J x K matrix yielding the total demand of households of type k for long-term housing in location j
    - EVs - (τ_bar*J) x K matrix of EV function values for each type
"""
function D_L(r::Vector{Float64},a::Matrix{Float64},P::NamedTuple,EVs_initial::Matrix{Float64}=rand(P.tau_bar*(P.J+1),P.K))
    D_Ls = zeros(Float64,P.J+1, P.K)
    EVs = zeros(Float64,P.tau_bar*(P.J+1),P.K)
    for k in 1:P.K
        Π, EV_G = trans_mat_EV(r,a,k,P,EVs_initial[:,k])
        stat_dist_k = stationary_dist_MC_iter(Π,P)
        stat_dist_all_k = reshape(stat_dist_k, P.tau_bar,P.J+1)
        D_Ls[:,k] = P.Pop[k]*sum(stat_dist_all_k,dims=1)'
        EVs[:,k] = EV_G
    end
    #println(D_Ls[:,1])
    return D_Ls, EVs
end





###################################################################################################
## 3. EQUILIBRIUM SOLVERS
###################################################################################################

####### Normalized excess demand 

function ED_L_norm(r::Vector{Float64},p::Vector{Float64},a::Matrix{Float64},P::NamedTuple,EVs_initial::Matrix{Float64}=rand(P.tau_bar*(P.J+1),P.K))
    # Get demand for living area conditional on living in a location
    DL,EVs_g = D_L(r,a,P,EVs_initial)
    DL_mat = DL[1:P.J,:]
    
    # Compute the total demand
    demand_sq_area = D_sq_area(r,P)
    DL_mat = DL_mat .* demand_sq_area

    # Aggregate across households by summing up to get the total demand
    D = DL_mat*ones(P.K)

    # Supply
    S = ((S_L(r,p,P,options) .* P.H) + P.H_LT) .* P.avg_squared_footage

    # Multiply the housing supply by the average squared footage in the area to get total squared footage available in the location
    Stat_ED_vec = (D - S)./(0.5*(D+S))

    return Stat_ED_vec
end


####### Normalized excess demand 

function ED_S_norm(r::Vector{Float64},p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)

    # Get tourist guest demand
    DG = D_S(r,p,a,P)

    # Transform to unit demand
    D = DG./P.listings_to_total_str_guests 

    # Supply
    S = (1 .- S_L(r,p,P,options)) .* P.H

    # Multiply the housing supply by the average squared footage in the area to get total squared footage available in the location
    Stat_ED_vec = (D - S)./(0.5*(D+S))*P.include_airbnb

    return Stat_ED_vec
end


"""
function tatonnement_prices(initial_r::Vector{Float64},initial_p::Vector{Float64},a::Matrix{Float64},P::NamedTuple,max_iter::Int64 = 10000,
    EVs_initial::Matrix{Float64}=ones(P.tau_bar*(P.J+1),P.K), δ::Float64 = 0.1, shrinkage::Float64 = 0.97,
    update_param_tol::Float64 = 10^-2,algo_tol::Float64 = 5*10^-8)
    For a given matrix of amenities and an initial guess of equilibrium prices, computes equilibrium in the housing market
    using the tatonnement adjustment algorithm in an adaptive way. The update parameter multiplying the excess demand vector
    shrinks whenever the objective function increases between steps. When the update parameter gets too small as specified by
    the user, the function proceeds to employing the Nelder-Mead algorithm, using the last recorded vector of prices from the
    tatonnement algorithm as the initial value.

    Inputs (selected):
    - δ - initial value for the output parameter on the ED vector in the tatonnement procedure
    - shrinkage - a multiplicative factor which  determines the new value of the output parameter the value of the objective
        function increases between steps
    - update_param_tol - tolerance on δ. When the value of the update parameter goes below this value, the tatonnement algorithm
        is terminated and we proceed with using the Nelder-Mead algorithm

    Outputs:
    - x_g - equilibrium price vector
    - δ_c - last value of the update parameter in the Tatonnement algorithm before it was terminated
"""

function tatonnement_r(initial_r::Vector{Float64},a::Matrix{Float64},P::NamedTuple,max_iter::Int64 = 10000,
    EVs_initial::Matrix{Float64}=ones(P.tau_bar*(P.J+1),P.K), δ_prices::Float64 = 0.01, algo_tol::Float64 = 5*10^-8)

    # Initialize variables
    x_g = copy(initial_r)
    iter = 0
    tol = 1000
    DL, EVs_g = D_L(x_g,a,P,EVs_initial)

    
    # Iterate until convergence
    while ((tol > algo_tol) && (iter < max_iter) )

        # Store last compute value
        x_old = copy(x_g)
        EV_old = copy(EVs_g)

        # Compute excess demand function
        EDL = ED_L_norm(x_old,P.p_observed,a,P,EV_old)
       
        # Update prices 
        EDL_adjusted = max.(δ_prices*EDL, -δ_prices*ones(P.J)).*(EDL.<0) + min.(δ_prices*EDL,δ_prices*ones(P.J)).*(EDL.>0)
        x_g = exp.(log.(x_old) + EDL_adjusted)

        # Update EVs
        DL,EVs_g = D_L(x_g,a,P,EVs_g)

        # Check convergence criterion
        tol = norm(EDL,Inf)

        # Iterate
        iter += 1
    end

    return x_g
end


function nested_tatonnement_r_p(initial_r::Vector{Float64},initial_p::Vector{Float64},initial_a::Matrix{Float64},P::NamedTuple,options)

    # Initialize variables
    a = initial_a
    r_g = copy(initial_r)
    p_g = copy(initial_p)
    iter_outer = 0
    iter_inner = 0
    tol_outer = 1000
    tol_inner = 1000
    EVs_g = copy(options.EVs_initial)
    DL_g = zeros(P.J,P.K)

    
    # Iterate until convergence
    while ((tol_outer > options.tol_r) && (iter_outer < options.max_iter_r) )

        # Solve for p 
        while ((tol_inner > options.tol_p) && (iter_inner < options.max_iter_p))
       # Store last compute value
        p_old = copy(p_g)

        # Compute step 
        EDS = ED_S_norm(r_g,p_g,a,P)
        EDS_adjusted = max.(options.δ_prices*EDS, -options.δ_prices*ones(P.J)).*(EDS.<0) + min.(options.δ_prices*EDS,options.δ_prices*ones(P.J)).*(EDS.>0)

        # Update prices
        p_g = ifelse(options.endogenous_p == true, exp.(log.(p_old) + EDS_adjusted), initial_p)
        
        # Check convergence criterion
        tol_inner = norm(p_g-p_old,Inf)

        iter_inner += 1

        end

        # Reset for next iteration
        iter_inner = 0
        tol_inner = 1000

        # Store last compute value
        r_old = copy(r_g)
        EVs_old = copy(EVs_g)
        DL_old = copy(DL_g)
        
        # Compute step 
        EDL = ED_L_norm(r_g,p_g,a,P)
        EDL_adjusted = max.(options.δ_prices*EDL, -options.δ_prices*ones(P.J)).*(EDL.<0) + min.(options.δ_prices*EDL,options.δ_prices*ones(P.J)).*(EDL.>0)

        # Update prices
        r_g = ifelse(options.endogenous_r == true, exp.(log.(r_old) + EDL_adjusted), initial_r)

        # Update EVs
        DL_g,EVs_g = D_L(r_g,a,P,EVs_old)
        DL_g = DL_g[1:P.J,:]

        # Check convergence criterion 
        tol_outer = norm(r_old - r_g,Inf)

        iter_outer += 1
    end

    println("Tatonnement prices converged after ", iter_outer, " iterations, with tolerance ", tol_outer)
    println("-----------------------------------------------------")
    return r_g, p_g, EVs_g, DL_g
end

"""
function full_solver(initial_a::Matrix{Float64},initial_r::Vector{Float64},initial_p::Vector{Float64}, P::NamedTuple)
    Main function for the nested fixed-point algorithm for equilibrium computation. The function allows user to specify the solution algorithm
        in the inner loop.
"""

function full_solver(initial_r::Vector{Float64},initial_p::Vector{Float64}, initial_a::Matrix{Float64},P::NamedTuple,options)

    # Initialize variables
    a_g = copy(initial_a)
    r_g = copy(initial_r)
    p_g = copy(initial_p)
    DL_g = Matrix{Float64}(zeros(P.J,P.K))
    EVs_g = copy(options.EVs_initial)
    n_iter = 0
    dist_a = 100

    #println("Initializing algorithm")

    # Iterate until convergence with specified tolerance on the algorithm in the inner loop
    while (dist_a > options.tol_a && n_iter < options.max_iter_a)

        # Store last compute value
        r_old = copy(r_g)
        p_old = copy(p_g)
        EVs_old = copy(EVs_g)
        a_old = copy(a_g)

        # Compute the equilibrium vector of prices and the update parameter in the inner loop
        r_g, p_g, EVs_g, DL_g = nested_tatonnement_r_p(r_old,p_old,a_old,P,options)

        # Update expected continuation values 
        options.EVs_initial = EVs_g

        # Compute the corresponding amenity supply
        a_new = Amenity_supply(r_g,p_g,a_old,DL_g,P)

        # Update amenities
        a_g = ifelse(options.endogenous_a == true, (1 - options.λ)*a_new + options.λ*a_old ,initial_a) 


        # Check the convergence criterion
        dist_a = norm(a_old - a_g,Inf)

        # Update counter
        n_iter += 1

        println("-----------------------------------------------------")
        println("Iteration number ", n_iter, " with tolerance ", dist_a)

        CSV.write("temp/r_endo_$(gamma_files)_loop.csv",  Tables.table(r_g), writeheader=false)
        CSV.write("temp/p_endo_$(gamma_files)_loop.csv",  Tables.table(p_g), writeheader=false)
        CSV.write("temp/a_endo_$(gamma_files)_loop.csv",  Tables.table(a_g), writeheader=false)

    end
    
    # Return equilibrium objects
    return r_g, p_g, a_g, EVs_g, DL_g
end

###################################################################################################
## 4. WELFARE
###################################################################################################

"""
function welfare_landlords(r::Vector{Float64},P::NamedTuple)
    Compute the welfare of landlord at rental prices r
"""
function welfare_landlords(r::Vector{Float64},p::Vector{Float64},P::NamedTuple)
    # Compute unit rent 
    r_annual = r .* P.avg_squared_footage

    # Compute Airbnb annual income
    p_annual = (1 - options.airbnb_tax_rate)*p*365

    # Compute welfare
    welfare_vec = log.(exp.(P.alpha.*r_annual/10^4) + exp.(P.alpha.*p_annual/10^4 - P.kappa_j))

    # Put in dollar terms
    welfare_locwise = mean(welfare_vec, weights(P.H_LT))*10^4/P.alpha

    return welfare_locwise
end

"""
function welfare_tourists(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    Compute the welfare of tourists 
"""
function welfare_tourists(p::Vector{Float64},a::Matrix{Float64},P::NamedTuple)
    p_guest = p./P.mean_accommodates
    y_hat_tourists = [[log.(p_guest);0] [log.(a); zeros(6)'] P.tourist_demand_controls]*P.θ_tourist
    δ_tourists = P.δ_j_tourist + y_hat_tourists
    δ_norm = δ_tourists[end-1]
    δ_tourists_norm = δ_tourists .- δ_norm
    E_tourists = exp.(δ_tourists_norm)
    welfare_tourists = log(sum(E_tourists))
    return welfare_tourists
end


"""
function welfare_households(r::Vector{Float64},a::Matrix{Float64},P::NamedTuple,EVs_initial::Matrix{Float64}=rand(P.tau_bar*(P.J+1),P.K)
    Computes welfare of each type of households for a given vector of prices and matrix of amenities.
"""
function welfare_households(r::Vector{Float64},a::Matrix{Float64},P::NamedTuple,EVs_initial::Matrix{Float64}=rand(P.tau_bar*(P.J+1),P.K))
    transition_mat = zeros((P.J+1)*P.tau_bar,(P.J+1)*P.tau_bar,P.K)
    EVs = zeros((P.J+1)*P.tau_bar,P.K)
    W = zeros(P.K)
    for k = 1:P.K
        Π, EV_k = trans_mat_EV(r,a,k,P,EVs_initial[:,k])
        transition_mat[:,:,k] = Π
        EVs[:,k] = EV_k
        W[k] = EV_k'*stationary_dist_MC_iter(transition_mat[:,:,k],P)
    end
    return W
end

"""
function CE_renter(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P,δ_r)
    Computes the consumption equivalent using the closed-form solution from the paper for renters 
"""
function CE_renter(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P,δ_r)
    W_0 =  welfare_households(r_orig,a_orig,P)
    W_1 =  welfare_households(r_eq,a_eq,P)
    muc = - (1/(1-P.beta)).*(δ_r./(P.income[1:P.K].*P.exp_shares[1:P.K]))
    CE =  (W_1-W_0)./muc
    return CE
end

"""
function CE_homeowner(r_orig::Vector{Float64},a_orig::Matrix{Float64},p_orig::Vector{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},p_eq::Vector{Float64},P,δ_r)
    Computes the consumption equivalent using the closed-form solution from the paper for homeowners 
"""
function CE_homeowner(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P,δ_r::Vector{Float64})
    
    # Welfare of renters
    CE = CE_renter(r_orig,a_orig,r_eq,a_eq,P,δ_r) 

    # Income from rents
    rental_income_orig = mean(P.avg_squared_footage.*r_orig, weights(P.H))
    rental_income_eq = mean(P.avg_squared_footage.*r_eq, weights(P.H))

    CS_homeowner = CE + (rental_income_eq-rental_income_orig)*ones(P.K)

    return CS_homeowner
end



####################################################################################################
####################################################################################################
################################### To be checked ##########################################
####################################################################################################
####################################################################################################


###################################################################################################
## 4. WELFARE
###################################################################################################


"""
function rent_equivalent(orig_welfare::Vector{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P, max_bounds = 10000*ones(P.K), min_bounds = 1*ones(P.K))
    Compute the rent equivalent w.r.t. orig_welfare and for new equilibrium prices and amenities, letting the households to re-sort according to this equilibrium.
    Uses a root finding method that requires the programmer to supply the minimum and maximum bounds on the root (the ren equivalent). Note that the rent equivalent is type-specific.
"""
function rent_equivalent(orig_welfare::Vector{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},P, max_bounds = 10000*ones(P.K), min_bounds = 1*ones(P.K))
    rent_equivs = zeros(P.K)

    for k in 1:P.K

        println(k)

        # Define the function we are going to set equal to the original welfare
        f(x) = welfare_households(r_eq + x*ones(P.J),a_eq,P)[k]

        # Define residual which we are going to minimize
        res(x) = f(x) - orig_welfare[k]
        rent_equivs[k] = find_zero(res, (-min_bounds[k]*minimum(r_eq)+10e-6,max_bounds[k]*maximum(r_eq)), Bisection())
    end

    return rent_equivs
end

"""
function consumption_equivalent(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},rental_income_orig::Vector{Float64},rental_income_eq::Vector{Float64},P)
    Computes the consumption equivalent using the closed-form solution from the paper.
"""
function consumption_equivalent(r_orig::Vector{Float64},a_orig::Matrix{Float64},r_eq::Vector{Float64},a_eq::Matrix{Float64},rental_income_orig::Vector{Float64},rental_income_eq::Vector{Float64},P,δ_r )
    W_0 =  welfare_households(r_orig,a_orig,P)
    W_1 =  welfare_households(r_eq,a_eq,P)
    CE = exp.((1 - P.beta)*abs.(((1 .- P.exp_shares[1:P.K])./δ_r )).*(W_0 - W_1)).*(P.income[1:P.K] + rental_income_orig) - P.income[1:P.K]- rental_income_eq
    return CE
end


###################################################################################################
## 5. SEGREGATION AND INEQUALITY
###################################################################################################

"""
function entropy_index_locations(D::Matrix{Float64},P::NamedTuple)
    Computes the entropy index for a single location, given a distribution matrix of each type across locations D (rows index locations, columns index types)
"""
function entropy_index_locations(D::Matrix{Float64},P::NamedTuple)
    D_j = D*ones(P.K)
    d = D ./ D_j
    return - (d .* log.(d))*ones(P.K)
end

"""
function entropy_index_city(D::Matrix{Float64},P::NamedTuple)
    Computes the entropy index for the whole city. Follows White, Michael J. "Segregation and diversity measures in population distribution." __Population index__ (1986): 198-221.
"""
function entropy_index_city(D::Matrix{Float64},P::NamedTuple)
    D_k = D'ones(P.J)
    D_j = D*ones(P.K)
    Dsum = sum(D)
    H_hat = - ((D_k/Dsum) .* log.(D_k/Dsum))'ones(P.K)

    H_bar = (entropy_index_locations(D,P) .* (D_j/Dsum))'ones(P.J)

    return (H_hat - H_bar) / H_hat
end

"""
function gini_coeff(data::Vector{<:Real})
    Computes the Gini coefficient for any allocation vector
"""
function gini_coeff(data::Vector{<:Real})
    @assert all(x->x >= 0, data)
    y = sort(data)
    n = length(y)
    # cargo-culted from Wikipedia
    2 * sum([i * y[i] for i in 1:n]) / (n * sum(y)) - (n + 1) / n
end

"""
function gini_weighted(datadistarray)
    Computes a weighted Gini coefficient. The input is a matrix where the second column are the supplied weights.
"""
function gini_weighted(datadistarray)
    if minimum(datadistarray[:,1]) < 0
        printstyled("There are negative input values - use a different inequality measure \n", bold=:true, color=:red)
        return
    else
        if size(datadistarray,2) == 1
            n = length(datadistarray)
        	datadistarray_sorted = sort(datadistarray)
        	2*(sum(collect(1:n).*datadistarray_sorted))/(n*sum(datadistarray_sorted))-1
        elseif size(datadistarray,2) == 2
            Swages = cumsum(datadistarray[:,1].*datadistarray[:,2])
            Gwages = Swages[1]*datadistarray[1,2] + sum(datadistarray[2:end,2] .* (Swages[2:end]+Swages[1:end-1]))
            return 1 - Gwages/Swages[end]
        else
            printstyled("Input data does not have conformable dimension \n", bold=:true, color=:red)
            return
        end
    end
end

