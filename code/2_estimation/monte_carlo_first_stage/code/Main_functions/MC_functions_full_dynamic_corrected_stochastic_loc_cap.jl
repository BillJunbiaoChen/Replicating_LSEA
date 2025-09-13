####################################################################################  
## Code: MC Helper Functions
## Authors: Marek Bojko & Sriram Tolety 
####################################################################################  

mutable struct Data_struct_RHS
    budget::Vector{Float64}
    budget_exo::Vector{Float64}
    budget_neg::Vector{Float64}
    amenities::Matrix{Float64}
    amenities_exo::Matrix{Float64}
    ξ::Vector{Float64}
    dist_mat::Matrix{Float64}
end

mutable struct Data_struct_observed
    Y::Vector{Float64}
    X::Matrix{Float64}
    Z::Matrix{Float64}
    which_indices_CCP::Vector{Float64}
    indices_unobserved::Vector{Float64}
    indices_observed::Vector{Float64}
end

function Γ_tensor(Pdims::NamedTuple,P_mat::Matrix{Float64})
    Gamma_tensor = zeros(Float64,Pdims.tau_bar,Pdims.J,Pdims.J,Pdims.tau_bar)
    for j_prime in 1:Pdims.J
        for j in 1:Pdims.J
            if j == j_prime
                Gamma_tensor[:,j,j_prime,:] = P_mat
            else
                Gamma_tensor[:,j,j_prime,1] = ones(Pdims.tau_bar)
            end
        end
    end
    return Gamma_tensor
end

function Γ_tensor_full_t(Pdims::NamedTuple,P_mat::Matrix{Float64})
    Gamma_tensor = zeros(Float64,Pdims.tau_bar,Pdims.J,Pdims.T,Pdims.J,Pdims.tau_bar)
    for j_prime in 1:Pdims.J
        for t in 1:Pdims.T
            for j in 1:Pdims.J
                if j == j_prime
                    Gamma_tensor[:,j,t,j_prime,:] = P_mat
                else
                    Gamma_tensor[:,j,t,j_prime,1] = ones(Pdims.tau_bar)
                end
            end
        end
    end
    return Gamma_tensor
end

function get_index_state(j::Int64,tau::Int64,Pdims::NamedTuple)
    indices = [(j_x,tau_x) for j_x in 1:Pdims.J for tau_x in 1:Pdims.tau_bar]
    return findall(x->x==(j,tau), indices)[1]
end

function normalize_mat(M::Matrix{Float64},Pdims::NamedTuple)
    v_outside_mat = M[(Pdims.J-1)*(Pdims.T)+1:end,:]
    v_outside_mat_stacked = kron(ones(Pdims.J),v_outside_mat)
    v_mat_norm = M - v_outside_mat_stacked
    return v_mat_norm
end

function compress_mat_loc_capital(M::Matrix{Float64},Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.T,Pdims.tau_bar,Pdims.J,Pdims.J*Pdims.tau_bar)
    M_tensor_tau_summed = sum(M_tensor,dims=2)
    M_mat_reduced = reshape(M_tensor_tau_summed,Pdims.T*Pdims.J,Pdims.J*Pdims.tau_bar)
    return M_mat_reduced
end

function reshape_mnl_compatible(M::Matrix{Float64},Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.T,Pdims.J,Pdims.J*Pdims.tau_bar)
    output_tensor = zeros(Pdims.T,Pdims.J*Pdims.tau_bar,Pdims.J)
    for t in 1:Pdims.T
        output_tensor[t,:,:] = M_tensor[t,:,:]'
    end
    output_matrix = reshape(output_tensor,Pdims.T*Pdims.J*Pdims.tau_bar,Pdims.J)
    return output_matrix
end

function reshape_mnl_to_model(M::Matrix{Float64},Pdims::NamedTuple)
    println(first(M, 5))
    M_tensor = reshape(M,Pdims.T,Pdims.J*Pdims.tau_bar,Pdims.J)
    output_tensor = zeros(Pdims.T,Pdims.J,Pdims.J*Pdims.tau_bar)
    for t in 1:Pdims.T
        output_tensor[t,:,:] = M_tensor[t,:,:]'
    end
    output_matrix = reshape(output_tensor,Pdims.T*Pdims.J,Pdims.J*Pdims.tau_bar)
    return output_matrix
end

function Y_next_j(j::Int64,log_CCP::Matrix{Float64},Pdims::NamedTuple)
    # Prepare results
    Y_tnext_j = zeros(Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T)

    log_CCP_tensor = reshape(copy(log_CCP),Pdims.T,Pdims.J,Pdims.tau_bar,Pdims.J)

    # Exclude 0 and j as the renewal actions, and shift the time periods one period backwards
    #log_CCP_tensor_reduced_j = log_CCP_tensor[:,[d for d in 1:Pdims.J if d ∉ [Pdims.J,j]],:,:]
    log_CCP_tensor_reduced_j = log_CCP_tensor
    log_CCP_tensor_reduced_j[1:end-1,:,:,:] = log_CCP_tensor_reduced_j[2:end,:,:,:]
    log_CCP_tensor_reduced_j[end,:,:,:] = zeros(Pdims.J,Pdims.tau_bar,Pdims.J)

    # Compose all Y's for j
    renewal_actions_j = [d for d in 1:Pdims.J if d!=j && d!= Pdims.J]

    for t in 1:Pdims.T
        for j_prev in 1:Pdims.J
            for tau_prev in 1:Pdims.tau_bar
                for (id_renewal,renewal_action) in enumerate(renewal_actions_j)
                    prob_dist_tau_j = Pdims.Gamma_tensor[tau_prev,j_prev,j,:]
                    prob_dist_tau_0 = Pdims.Gamma_tensor[tau_prev,j_prev,Pdims.J,:]
                    trans_probs_next_j = log_CCP_tensor_reduced_j[t,renewal_action,:,j]
                    trans_probs_next_0 = log_CCP_tensor_reduced_j[t,renewal_action,:,Pdims.J]
                    Y_tnext_j[id_renewal,tau_prev,j_prev,t] = prob_dist_tau_j'*trans_probs_next_j - prob_dist_tau_0'*trans_probs_next_0;
                end
            end
        end
    end

    # Reshape to a vector indexed by (j_prev,tau_prev,j_tilde) for j_tilde a renewal action
    Y_next_vec_j = reshape(Y_tnext_j,(Pdims.J-2)*Pdims.tau_bar*Pdims.J*Pdims.T)
    return Y_next_vec_j
end

function Y_next(log_CCP::Matrix{Float64},Pdims::NamedTuple)
    Y_next_j_vecs = [Y_next_j(j,copy(log_CCP),Pdims) for j in 1:Pdims.J-1]
    return vcat(Y_next_j_vecs...)
end

function dist_est_j_old(j::Int64,dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    dist_j_norm = dist_mat[:,j] - dist_mat[:,Pdims.J]
    dist_j_norm_prev = repeat(dist_j_norm,outer=Pdims.T)
    dist_j_norm_prev = repeat(dist_j_norm_prev,inner = Pdims.tau_bar*(Pdims.J-2))
    dist_j_norm_next = repeat(dist_j_norm[[d for d in 1:Pdims.J if d != j && d!= Pdims.J]],outer = Pdims.T*Pdims.tau_bar*Pdims.J)
    return Pparam.ρ*dist_j_norm_next + dist_j_norm_prev
end

function dist_est_old(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    dist_est_vecs = [dist_est_j(j,dist_mat,Pdims,Pparam) for j in 1:Pdims.J-1]
    return vcat(dist_est_vecs...)
end

function dist_est_j(j::Int64,dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    # Define useful auxiliary matrices
    ones_above_j_neq_k = ones(Pdims.tau_bar*(Pdims.J-2)*(j-1))
    ones_below_j_neq_k = ones(Pdims.tau_bar*(Pdims.J-2)*(Pdims.J-j))
    zeros_j_neq_k = zeros(Pdims.tau_bar*(Pdims.J-2))
    full_vec_j_neq_k = repeat(vcat([ones_above_j_neq_k,zeros_j_neq_k,ones_below_j_neq_k]...), outer = Pdims.T)

    ones_above_0_neq_k = ones(Pdims.tau_bar*(Pdims.J-2)*(Pdims.J-1))
    zeros_0_neq_k = zeros(Pdims.tau_bar*(Pdims.J-2))
    full_vec_0_neq_k = repeat([ones_above_0_neq_k ; zeros_0_neq_k], outer = Pdims.T)

    # Column coming from gamma_0
    gamma_0_col = Pparam.ρ*ones(Pdims.T*Pdims.J*Pdims.tau_bar*(Pdims.J-2)) + full_vec_j_neq_k .* full_vec_0_neq_k

    # Column coming from gamma_1
    dist_from_j = vec(dist_mat[:,j])
    dist_j_renewal_vec = [dist_from_j[1:j-1]; dist_from_j[j+1:Pdims.J-1]]
    dist_j_renewal_full_vec = repeat(dist_j_renewal_vec,outer = Pdims.T*Pdims.J*Pdims.tau_bar)
    full_dist_vec = [vec(dist_mat[:,j]);1000]
    dist_from_j_full_mT = repeat(full_dist_vec,inner = Pdims.tau_bar*(Pdims.J-2))
    dist_from_j_full = repeat(dist_from_j_full_mT,outer = Pdims.T)
    gamma_1_col = Pparam.ρ*dist_j_renewal_full_vec + (dist_from_j_full .* full_vec_j_neq_k) .* full_vec_0_neq_k

    # Column coming from gamma_2
    gamma_2_col =  - Pparam.ρ*ones(Pdims.T*Pdims.J*Pdims.tau_bar*(Pdims.J-2)) - full_vec_0_neq_k + (1 .- full_vec_0_neq_k)

    return [gamma_0_col gamma_1_col gamma_2_col]
    #return gamma_1_col
end

function dist_est(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    dist_est_vecs = [dist_est_j(j,dist_mat,Pdims,Pparam) for j in 1:Pdims.J-1]
    return vcat(dist_est_vecs...)
end

function dist_est_j_data_gen(j::Int64,dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    # Define useful auxiliary objects
    ones_j_neq_k = ones(Pdims.J); ones_j_neq_k[j] = 0
    ones_k_neq_0 = ones(Pdims.J); ones_k_neq_0[Pdims.J] = 0

    # Column coming from gamma_0
    gamma_0_col = Pparam.gamma_0 * (ones_k_neq_0 .* ones_j_neq_k)

    # Column coming from gamma_1
    gamma_1_col = Pparam.gamma_1*([vec(dist_mat[:,j]);0])

    # Column coming from gamma_2
    gamma_2_col = Pparam.gamma_2*(1 .- ones_k_neq_0)

    #return gamma_0_col + gamma_1_col + gamma_2_col
    return gamma_1_col + gamma_2_col
end

#=
function dist_est_data_gen(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    dist_est_vecs = [dist_est_j_data_gen(j,dist_mat,Pdims,Pparam) for j in 1:Pdims.J-1]
    dist_est_vec_outer = Pparam.gamma_2*[ones(Pdims.J-1);0]
    append!(dist_est_vecs,[dist_est_vec_outer])
    return hcat(dist_est_vecs...)
end
=#

function dist_est_data_gen(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    ones_off_diag = ones(Pdims.J-1,Pdims.J-1) - diagm(ones(Pdims.J-1))
    inner_mat = Pparam.gamma_0 * ones_off_diag  + Pparam.gamma_1 * dist_mat
    #inner_mat =  Pparam.gamma_1 * dist_mat
    #inner_mat = zeros(Pdims.J-1,Pdims.J-1)
    full_mat_h = hcat(inner_mat, Pparam.gamma_2 * ones(Pdims.J-1))
    full_mat = vcat(full_mat_h, Pparam.gamma_2 *[ones(Pdims.J-1);0]')
    #full_mat_h = hcat(inner_mat, zeros(Pdims.J-1))
    #full_mat = vcat(full_mat_h, zeros(Pdims.J)')
    return full_mat
end

function remove_T_vec(v::Vector{Float64},Pdims::NamedTuple)
    v_mat = reshape(v,(Pdims.J-2)*Pdims.J*Pdims.tau_bar,Pdims.T,Pdims.J-1)
    v_mat_reduced = v_mat[:,1:end-1,:]
    v_reduced = reshape(v_mat_reduced,(Pdims.T-1)*(Pdims.J-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    return v_reduced
end

function remove_T_mat(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,(Pdims.J-2)*Pdims.J*Pdims.tau_bar,Pdims.T,Pdims.J-1,size(M)[2])
    M_tensor_reduced = M_tensor[:,1:end-1,:,:]
    M_reduced = reshape(M_tensor_reduced,(Pdims.T-1)*(Pdims.J-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2),size(M)[2])
    return M_reduced
end


function DGP_RHS(Pdims::NamedTuple,Pparam::NamedTuple)
    # generate xi
    d_u = Normal(Pparam.μ_u, Pparam.σ_u)
    u = [rand(d_u,(Pdims.J-1)*Pdims.T);zeros(Pdims.T)] #J*T*tau_bar vector
    d_v = Normal(Pparam.μ_v, Pparam.σ_v)
    v = [rand(d_v,(Pdims.J-1)*Pdims.T);zeros(Pdims.T)] #J*T vector
    ξ = u + v

    # generate budget
    d_budget = LogNormal(Pparam.μ_b, Pparam.σ_b)
    #d_budget = Normal(0, Pparam.σ_b)
    budget_exo = [rand(d_budget, (Pdims.J-1)*Pdims.T);ones(Pdims.T)]
    budget = Pparam.κ*budget_exo + (1-Pparam.κ)*v
    budget_neg = replace(x -> x <= 0 ? 1 : 0, budget)
    replace!(x -> x<= 0 ? 10^-10 : x, budget)

    # generate amenities
    d_amenity = LogNormal(Pparam.μ_a, Pparam.σ_a)
    amenities_exo = rand(d_amenity, (Pdims.J-1)*Pdims.T,Pdims.S)
    amenities_exo = vcat(amenities_exo,ones(Pdims.T,Pdims.S)) # (J*T)xS matrix
    amenities = Pparam.η*amenities_exo .+ (1-Pparam.η)*v

    ## generate moving costs
    # First, generate a distance matrix
    d_distance = LogNormal(Pparam.μ_d,Pparam.σ_d)
    D_r = rand(d_distance,Pdims.J-1,Pdims.J-1)
    dist_mat_inner = D_r'*D_r

    # Set diagonal to 0
    dist_mat = dist_mat_inner - Diagonal(diag(dist_mat_inner))

    D_RHS = Data_struct_RHS(budget,budget_exo,budget_neg,amenities,amenities_exo,ξ,dist_mat)

    return D_RHS
end



"""
Older function where we generated distances to the outside option
"""
function DGP_RHS_full_dist_mat(Pdims::NamedTuple,Pparam::NamedTuple)
    # generate xi
    d_u = Normal(Pparam.μ_u, Pparam.σ_u)
    u = [rand(d_u,(Pdims.J-1)*Pdims.T);zeros(Pdims.T)] #J*T*tau_bar vector
    d_v = Normal(Pparam.μ_v, Pparam.σ_v)
    v = [rand(d_v,(Pdims.J-1)*Pdims.T);zeros(Pdims.T)] #J*T vector
    ξ = u + v

    # generate budget
    d_budget = LogNormal(Pparam.μ_b, Pparam.σ_b)
    #d_budget = Normal(0, Pparam.σ_b)
    budget_exo = [rand(d_budget, (Pdims.J-1)*Pdims.T);ones(Pdims.T)]
    budget = Pparam.κ*budget_exo + (1-Pparam.κ)*v
    budget_neg = replace(x -> x <= 0 ? 1 : 0, budget)
    replace!(x -> x<= 0 ? 10^-10 : x, budget)

    # generate amenities
    d_amenity = LogNormal(Pparam.μ_a, Pparam.σ_a)
    amenities_exo = rand(d_amenity, (Pdims.J-1)*Pdims.T,Pdims.S)
    amenities_exo = vcat(amenities_exo,ones(Pdims.T,Pdims.S)) # (J*T)xS matrix
    amenities = Pparam.η*amenities_exo .+ (1-Pparam.η)*v

    ## generate moving costs
    # First, generate a distance matrix
    d_distance = LogNormal(Pparam.μ_d,Pparam.σ_d)
    D_r = rand(d_distance,Pdims.J-1,Pdims.J-1)
    dist_mat_inner = D_r'*D_r
    max_dist_inner = maximum(dist_mat_inner)
    distances_to_outside = max_dist_inner*ones(Pdims.J-1) + rand(d_distance,Pdims.J-1)
    dist_mat = vcat(hcat(dist_mat_inner,distances_to_outside),[distances_to_outside;0]')

    # Set diagonal to 0
    dist_mat = dist_mat - Diagonal(diag(dist_mat))

    D_RHS = Data_struct_RHS(budget,budget_exo,budget_neg,amenities,amenities_exo,ξ,dist_mat)

    return D_RHS
end

function gen_loc_cap_mats(Pdims::NamedTuple)
    # Generate matrix giving the evolution of location capital - indicator matrix
    mat_diff_loc = vcat(ones(Int64,1,Pdims.tau_bar),zeros(Int64,Pdims.tau_bar-1,Pdims.tau_bar))
    mat_same_loc = vcat(zeros(Int64,1,Pdims.tau_bar),hcat(Matrix{Int64}(I,Pdims.tau_bar-1,Pdims.tau_bar-1),[zeros(Int64,Pdims.tau_bar-2);1]))
    tau_mat_ind = vcat([[repeat(mat_diff_loc,outer = (1,j-1)) mat_same_loc  repeat(mat_diff_loc,outer = (1,Pdims.J-j))] for j in 1:Pdims.J]...) #(J*tau_bar) x (J*tau_ba) matrix
    tau_mat_ind_full = kron(tau_mat_ind,ones(Pdims.T)) #(J*tau_bar**T) x (J*tau_bar) matrix

    # Generate matrix giving the evolution of location capital - resulting loc capital
    row_diff_loc = ones(Pdims.tau_bar)
    row_same_loc = [collect(2:Pdims.tau_bar);Pdims.tau_bar]
    tau_mat = vcat([[repeat(row_diff_loc,outer = (j-1));row_same_loc;repeat(row_diff_loc,outer = (Pdims.J-j))]' for j in 1:Pdims.J]...) #J x (J*tau_bar) matrix
    tau_mat_full = kron(tau_mat,ones(Pdims.T)) #(J*T) x (J*tau_bar) matrix

    return tau_mat_ind_full, tau_mat_full, tau_mat
end

function invert_indices_matrix(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T-1,Pdims.J-1,size(M)[2])
    inverted_M_tensor = zeros(Pdims.T-1,Pdims.J-1,Pdims.J-2,Pdims.tau_bar,Pdims.J,size(M)[2])
    for j_init in 1:Pdims.J
        for tau_init in 1:Pdims.tau_bar
            for j_renewal in 1:Pdims.J-2
                inverted_M_tensor[:,:,j_renewal,tau_init,j_init,:] = M_tensor[j_renewal,tau_init,j_init,:,:,:]
            end
        end
    end
    inverted_M_mat = reshape(inverted_M_tensor,(Pdims.T-1)*(Pdims.J-1)*(Pdims.J-2)*Pdims.tau_bar*Pdims.J,size(M)[2])
    return inverted_M_mat
end


function invert_indices_vector(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T-1,Pdims.J-1)
    inverted_M_tensor = zeros(Pdims.T-1,Pdims.J-1,Pdims.J-2,Pdims.tau_bar,Pdims.J)
    for j_init in 1:Pdims.J
        for tau_init in 1:Pdims.tau_bar
            for j_renewal in 1:Pdims.J-2
                inverted_M_tensor[:,:,j_renewal,tau_init,j_init] = M_tensor[j_renewal,tau_init,j_init,:,:]
            end
        end
    end
    inverted_M_mat = reshape(inverted_M_tensor,(Pdims.T-1)*(Pdims.J-1)*(Pdims.J-2)*Pdims.tau_bar*Pdims.J)
    return inverted_M_mat
end


function DEV(EV::Vector{Float64},Pdims::NamedTuple)
    EV_mat = reshape(EV,Pdims.tau_bar,Pdims.J)
    EV_mat_rep = repeat(EV_mat', inner = (Pdims.J*Pdims.tau_bar,1))
    Gamma_mat = reshape(Pdims.Gamma_tensor,Pdims.J*Pdims.tau_bar*Pdims.J,Pdims.tau_bar)
    DEV_mat_full = Gamma_mat .* EV_mat_rep
    DEV_mat_avg = DEV_mat_full*ones(Pdims.tau_bar)
    DEV_mat = reshape(DEV_mat_avg,Pdims.J*Pdims.tau_bar,Pdims.J)
    return DEV_mat'
end

function EV_stationary(u_period::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    # Initialize
    EV_G = rand(Pdims.J*Pdims.tau_bar)
    dist_func = 1
    iter = 0

    # Iterate
    while dist_func > Pparam.tol && iter <= Pparam.max_iter
        # Deal with potential overflow issues
        EV_G_max = max(maximum(EV_G),1)
        DEV_mat = DEV(EV_G,Pdims)
        v_transpose = u_period + Pparam.ρ*DEV_mat .- EV_G_max
        v = v_transpose'
        EV = log.(exp.(v)*ones(Pdims.J)) .+ EV_G_max
        dist_func = norm(EV-EV_G,Inf)
        EV_G = copy(EV)
        iter += 1
    end
    return EV_G
end

function EV_t(u_period_next::Matrix{Float64},EV_next::Vector{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    DEV_mat = DEV(EV_next,Pdims)
    v_transpose = u_period_next + Pparam.ρ*DEV_mat
    v = v_transpose'
    EV_current = log.(exp.(v)*ones(Pdims.J))
    return EV_current
end

function trans_prob_single_t(u_period::Matrix{Float64},EV::Vector{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    EV_max = max(maximum(EV),1)
    DEV_mat = DEV(EV,Pdims)
    v = u_period + Pparam.ρ*DEV_mat .- EV_max # columns indexed by (j_init,tau_init) and rows by j
    denom = ones(Pdims.J)'*exp.(v)
    Q_mat = exp.(v) ./ denom  # columns indexed by (j_init,tau_init) and rows by j
    Q = reshape(Q_mat,Pdims.J,Pdims.tau_bar,Pdims.J)

    # Compute a transition tensor
    Pi_tensor = zeros(Float64,Pdims.tau_bar,Pdims.J,Pdims.tau_bar,Pdims.J)
    for tau in 1:Pdims.tau_bar
        for j in 1:Pdims.J
            for j_prime in 1:Pdims.J
                Pi_tensor[:,j_prime,tau,j] = Pdims.Gamma_tensor[tau,j,j_prime,:]*Q[j_prime,tau,j]
            end
        end
    end

    # Matricize
    Pi_mat = reshape(Pi_tensor,Pdims.tau_bar*Pdims.J,Pdims.tau_bar*Pdims.J)
    return Pi_mat
end


function gen_trans_probs(λ_t,λ_j,Pdims::NamedTuple,Pparam::NamedTuple,D_RHS)
    #### Unpack basic RHS variables
    # Unpack basic RHS variables
    ξ = D_RHS.ξ

    budget = D_RHS.budget
    budget_neg = D_RHS.budget_neg
    log_budget = log.(budget)

    amenities = D_RHS.amenities
    log_amenities = log.(amenities)

    budget_exo = D_RHS.budget_exo
    amenities_exo = D_RHS.amenities_exo

    dist_mat = D_RHS.dist_mat

    # Generate moving costs
    dist_mat_full = dist_est_data_gen(dist_mat,Pdims,Pparam)
    dist_mat_full = repeat(dist_mat_full, inner = (1,Pdims.tau_bar))
    dist_mat_full = repeat(dist_mat_full, inner = (Pdims.T,1))

    # Extend to the full set of state variables
    #dist_mat_full = repeat(dist_mat,inner=(1,Pdims.tau_bar))

    # Now extend to the full time range
    #dist_mat_full = repeat(dist_mat_full,inner = (Pdims.T,1))

    # Generate matrix giving the expected evolution of location capital
    Gamma_mat_all = reshape(Pdims.Gamma_tensor,Pdims.J*Pdims.tau_bar*Pdims.J,Pdims.tau_bar)
    if Pdims.include_tau_dummy == false
        Gamma_mat_avg = Gamma_mat_all*(Pparam.δ*collect(1:Pdims.tau_bar))
    else
        Gamma_mat_avg = Gamma_mat_all*[0;Pparam.δ]
    end
    Gamma_mat = reshape(Gamma_mat_avg,Pdims.J*Pdims.tau_bar,Pdims.J)
    tau_mat_full = repeat(Gamma_mat',inner = (Pdims.T,1))

    #### Compute conditional value function
    ## Generate flow utility - a J*tau_bar*T*J*tau_bar matrix;l rows are indexed by (j,tau,t), columns by (j_init,tau_init)
    if Pdims.FEs == false
        u = ξ .+ Pparam.α*log_budget .+ log_amenities*Pparam.β .+ dist_mat_full .+ tau_mat_full
    else
        u = ξ .+ Pparam.α*log_budget .+ log_amenities*Pparam.β .+ dist_mat_full .+ tau_mat_full .+ loc_dummy_mat*λ_j .+ time_dummy_mat*λ_t
    end

    u_tensor = reshape(u,Pdims.T,Pdims.J,Pdims.J*Pdims.tau_bar)

    #### Compute the EV functions and transition probabilities
    # Initialize
    trans_probs_tensor = zeros(Pdims.T,Pdims.J*Pdims.tau_bar,Pdims.J*Pdims.tau_bar)

    # First, start in the last period T and compute the "stationary" EV function
    u_T_tensor = u_tensor[Pdims.T,:,:]
    u_T = reshape(u_T_tensor,Pdims.J,Pdims.J*Pdims.tau_bar)
    EV_stat = EV_stationary(u_T,Pdims,Pparam)

    # Compute the implied transition matrix
    trans_probs_tensor[Pdims.T,:,:] = trans_prob_single_t(u_T,EV_stat,Pdims,Pparam)

    # Next, iterate backwards
    EV_next = EV_stat
    for t in Pdims.T-1:-1:1
        EV_current = EV_t(u_tensor[t+1,:,:],EV_next,Pdims,Pparam)
        trans_probs_tensor[t,:,:] = trans_prob_single_t(u_tensor[t,:,:],EV_current,Pdims,Pparam)
        EV_next = copy(EV_current)
    end

    # Reshape the tensor of transition probabilities to matrix
    trans_probs_mat = reshape(trans_probs_tensor,Pdims.J*Pdims.tau_bar*Pdims.T,Pdims.J*Pdims.tau_bar)
    return trans_probs_mat
end

function empirical_choices(prob_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    # Define indices
    indices = [(j_x,tau_x) for j_x in 1:Pdims.J for tau_x in 1:Pdims.tau_bar]

    # Compose the initial distribution
    s_0 = Pdims.initial_ind_state

    # Transform probability matrix into a tensor
    prob_tensor = reshape(prob_mat,Pdims.T,Pdims.J*Pdims.tau_bar,Pdims.J*Pdims.tau_bar)

    # create a tensor of transition flows where we will store the simulated choices
    choices_tensor = zeros(Pdims.T,Pdims.Pop,1,1)

    # Create a dataframe
    df_choices = DataFrame(t = Int64[], ind = Int64[], tau_init = Int64[], j_init = Int64[],
                                tau_prime = Int64[], j_prime = Int64[])

    # Loop through individuals, years, and locations
    for i in 1:Pdims.Pop
        s_0_i = s_0[i]
        for t in 1:Pdims.T
            # Compose the probability distribution to draw from
            prob_dist_i = vec(prob_tensor[t,:,s_0_i])
            prob_dist_i_mat = reshape(prob_dist_i,Pdims.tau_bar,Pdims.J)
            prob_dist_i_locs = sum(prob_dist_i_mat,dims=1)
            prob_dist_i_locs_vec = reshape(prob_dist_i_locs,Pdims.J)

            # For each agent, draw choice for the current period
            choice_i = sample(1:(Pdims.J*Pdims.tau_bar),Weights(prob_dist_i),1)
            choice_i = choice_i[1]

            # Compute choices and prepare to output
            tau_int = indices[s_0_i][2]
            j_int = indices[s_0_i][1]

            tau_p = indices[choice_i][2]
            j_p = indices[choice_i][1]

            # Push to dataframe
            el = [t,i,tau_int,j_int,tau_p,j_p]
            push!(df_choices,el)

            s_0_i = choice_i
        end
    end
    return df_choices
end

function get_frequency_probs(ind_choices_df::DataFrame,Pdims::NamedTuple)
    emp_choices_counts = zeros(Pdims.T,Pdims.tau_bar,Pdims.J,Pdims.J)
    emp_choices_probs = zeros(Pdims.T,Pdims.tau_bar,Pdims.J,Pdims.J)
    for t in 1:Pdims.T
        for j_init in 1:Pdims.J
            for tau_init in 1:Pdims.tau_bar
                for j in 1:Pdims.J
                    # Subset dataframe of individual choices
                    current_subset_ind_choices = subset(ind_choices_df,:t => ByRow(==(t)), :j_init => ByRow(==(j_init)), :tau_init => ByRow(==(tau_init)), :j_prime => ByRow(==(j)))
                    n_rows_current = nrow(current_subset_ind_choices)
                    emp_choices_counts[t,tau_init,j_init,j] = n_rows_current
                end
            end
        end
    end
    # Replace zero counts
    replace!(x -> x == 0 ? 10^-6 : x, emp_choices_counts)

    # Normalize
    denom = sum(emp_choices_counts,dims=4)
    emp_choices_probs = emp_choices_counts ./ denom

    # Reshape to matrix
    emp_choices_mat = reshape(emp_choices_probs,Pdims.T*Pdims.tau_bar*Pdims.J,Pdims.J)

    # Return as a dataframe
    emp_choices_df = DataFrame(emp_choices_mat,:auto)
    indices_all = [(j_x,tau_x,t_x) for j_x in 1:Pdims.J for tau_x in 1:Pdims.tau_bar for t_x in 1:Pdims.T]
    emp_choices_df[!,"t"] = [indic[3] for indic in indices_all]
    emp_choices_df[!,"tau_init"] = [indic[2] for indic in indices_all]
    emp_choices_df[!,"j_init"] = [indic[1] for indic in indices_all]

    return emp_choices_df
end

function find_states(vec_inds::AbstractArray{T},Pdims::NamedTuple) where T
    all_rows_inds = [(el[1],el[2],x,y,el[3]) for el in vec_inds if el[3] != Pdims.T for x in 1:(Pdims.J-2) for y in 1:(Pdims.J-1)]
    all_indices = [(j_init,tau_init,j_renewal,j,t) for j_init in 1:Pdims.J for tau_init in 1:Pdims.tau_bar for j_renewal in 1:Pdims.J-2 for j in 1:Pdims.J-1 for t in 1:Pdims.T-1]
    all_rows_pos = findfirst.(isequal.(all_rows_inds), (all_indices,))
    sort!(all_rows_pos)
    return all_rows_pos
end

function prepare_data(dat::DataFrame,CCP_df::DataFrame,dist_mat::Matrix{Float64},
                    Pdims::NamedTuple,Pparam::NamedTuple)

    #### Unpack basic RHS variables
    budget = vec(dat[!,"budget"])
    log_budget = log.(budget)

    amenities = Matrix(dat[!,r"am"])
    log_amenities = log.(amenities)

    budget_exo = vec(dat[!,"budget_exo"])
    amenities_exo = Matrix(dat[!,r"a_exo"])

    # Generate matrix giving the evolution of location capital
    tau_mat_inf_full, tau_mat_full, tau_mat = gen_loc_cap_mats(Pdims)

    #### Compute Y
    ## Prepare the transition matrix
    # Find which indices are missing (if any)
    CCP_j_ind = vec(CCP_df[!,"j_init"])
    CCP_tau_ind = vec(CCP_df[!,"tau_init"])
    CCP_t_ind = vec(CCP_df[!,"t"])
    indices_CCP = [[CCP_j_ind[i],CCP_tau_ind[i],CCP_t_ind[i]] for i in 1:length(CCP_t_ind)]
    indices_CCP_reduced = [ind_CCP for ind_CCP in indices_CCP if ind_CCP[3] != Pdims.T]
    indices_all = [[j_x,tau_x,t_x] for j_x in 1:Pdims.J for tau_x in 1:Pdims.tau_bar for t_x in 1:Pdims.T]
    which_indices_CCP = findfirst.(isequal.(indices_CCP_reduced), (indices_all,))
    ic = [x for x ∈ indices_all if x ∉ indices_CCP]

    # Fill in the unobserved states with small but positive transition probabilities
    for el in ic
        push!(CCP_df,vcat(el,Pparam.fill_prob*ones(Pdims.J)))
    end

    # Resort the dataframe
    sort!(CCP_df,[:j_init,:tau_init,:t])
    println(first(CCP_df, 5))
    CCP = Matrix{Float64}(CCP_df[!,r"p|x"])
    println(first(CCP, 5))

    # Compose CCPs
    CCP = reshape_mnl_to_model(CCP,Pdims)
    #CCP = compress_mat_loc_capital(CCP,Pdims)

    ## Rearrange CCPs to compute Y
    log_CCP = log.(CCP)
    log_CCP_norm = normalize_mat(copy(log_CCP),Pdims)
    log_CCP_norm_noout = log_CCP_norm[1:(Pdims.J-1)*Pdims.T,:]
    log_CCP_norm_noout_vec = reshape(log_CCP_norm_noout',(Pdims.J-1)*Pdims.J*Pdims.tau_bar*Pdims.T) # (j_prime,t,j,tau)
    log_CCP_norm_noout_full = repeat(log_CCP_norm_noout_vec,inner = Pdims.J-2)
    log_CCP_next_norm_full = Y_next(copy(log_CCP),Pdims) # (j_prime,t,j,tau)
    Y = log_CCP_norm_noout_full + Pparam.ρ*log_CCP_next_norm_full # Indexed outer->inner as: (j_prime,t,j_init,tau_init,j_renewal)

    #### Compose the RHS for the difference in utilities
    ## Moving costs based on distance
    dist_mat_full = dist_est(dist_mat,Pdims,Pparam)

    ## Location capital
    Gamma_mat_all = reshape(Pdims.Gamma_tensor_full_t,Pdims.J*Pdims.tau_bar*Pdims.T*Pdims.J,Pdims.tau_bar)
    if Pdims.include_tau_dummy == false
        Gamma_mat_avg = Gamma_mat_all*collect(1:Pdims.tau_bar)
    else
        Gamma_mat_avg = Gamma_mat_all*[0,1]
    end
    Gamma_mat = reshape(Gamma_mat_avg,Pdims.J*Pdims.tau_bar*Pdims.T,Pdims.J)

    # Normalize Gamma_mat by columns
    Gamma_mat_norm = Gamma_mat .- Gamma_mat[:,Pdims.J]

    # Reshape to stacked vectors after removing the outside option
    Gamma_mat_norm_noout = Gamma_mat_norm[:,1:Pdims.J-1]
    Gamma_mat_norm_noout_T = reshape(Gamma_mat_norm_noout,Pdims.J*Pdims.tau_bar,(Pdims.J-1)*Pdims.T)
    Gamma_mat_norm_noout_vec = reshape(Gamma_mat_norm_noout_T,(Pdims.J-1)*Pdims.J*Pdims.tau_bar*Pdims.T)

    # Repeat J-2 times to since this is the same for all renewal actions
    tau_vec_full = repeat(Gamma_mat_norm_noout_vec,inner=Pdims.J-2)

    ## Extend other variables
    log_budget_noout = log_budget[1:(Pdims.J-1)*Pdims.T]
    log_budget_full = repeat(log_budget_noout,inner=Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    log_amenities_noout = log_amenities[1:(Pdims.J-1)*Pdims.T,:]
    log_amenities_full = repeat(log_amenities_noout,inner=(Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
    budget_exo_noout = budget_exo[1:(Pdims.J-1)*Pdims.T]
    budget_exo_full = repeat(budget_exo_noout,inner = Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    amenities_exo_noout = amenities_exo[1:(Pdims.J-1)*Pdims.T,:]
    amenities_exo_full = repeat(amenities_exo_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
    loc_dummy_mat_noout = loc_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
    loc_dummy_mat_full = repeat(loc_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))
    time_dummy_mat_noout = time_dummy_mat[1:(Pdims.J-1)*Pdims.T,:]
    time_dummy_mat_full = repeat(time_dummy_mat_noout, inner = (Pdims.J*Pdims.tau_bar*(Pdims.J-2),1))

    #### Compose variables
    #X = [log_budget_full log_amenities_full dist_vec_full tau_vec_full Dm_full]
    X = [log_budget_full log_amenities_full dist_mat_full tau_vec_full]
    replace!(x -> x <= 0 ? 10^-10 : x, budget_exo_full)
    #Z = [log.(budget_exo_full) log.(amenities_exo_full) dist_vec_full tau_vec_full Dm_full]
    Z = [log.(budget_exo_full) log.(amenities_exo_full) dist_mat_full tau_vec_full]

    #### Remove the last time period as we are not able to identify it
    Y_reduced = remove_T_vec(Y,Pdims)
    X_reduced = remove_T_mat(X,Pdims)
    Z_reduced = remove_T_mat(Z,Pdims)

    #### Invert indices - get (j_init,tau_init,j_renewal,j,t) for rows; columns stay the same
    Y_reduced_inverted = invert_indices_vector(Y_reduced,Pdims)
    X_reduced_inverted = invert_indices_matrix(X_reduced,Pdims)
    Z_reduced_inverted = invert_indices_matrix(Z_reduced,Pdims)

    #### Remove missing states
    indices_unobserved = find_states(ic,Pdims)
    indices_observed = find_states(indices_CCP,Pdims)
    #Y_reduced_inverted[Int.(indices_unobserved)] = zeros(length(indices_unobserved))
    #X_reduced_inverted[Int.(indices_unobserved),:] = zeros(length(indices_unobserved),size(X)[2])
    #Z_reduced_inverted[Int.(indices_unobserved),:] = zeros(length(indices_unobserved),size(Z)[2])

    #X_reduced = [X_reduced Dm_full]
    #Z_reduced = [Z_reduced Dm_full]

    D = Data_struct_observed(Y_reduced_inverted,X_reduced_inverted,Z_reduced_inverted,Int.(which_indices_CCP),Int.(indices_unobserved),Int.(indices_observed))
end

function project(v::AbstractVector{T},P_mat::AbstractMatrix{R},Pdims) where {T,R}
    v_blocks = reshape(v,(Pdims.J-1)*(Pdims.T-1),Pdims.J*Pdims.tau_bar*(Pdims.J-2))

    # Compute the block projection
    proj_block = zeros((Pdims.J-1)*(Pdims.T-1))
    for k in 1:Pdims.J*Pdims.tau_bar*(Pdims.J-2)
        proj_block = proj_block + P_mat*v_blocks[:,k]
    end
    proj_block = 1/(Pdims.J*Pdims.tau_bar*(Pdims.J-2))*proj_block

    # Stack up to a vector
    proj_block_stacked = repeat(proj_block, outer=Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    return proj_block_stacked
end

function ols_mat(v::AbstractVector{T},P_OLS::AbstractMatrix{R},Pdims) where {T,R}
    v_blocks = reshape(v,(Pdims.J-1)*(Pdims.T-1),Pdims.J*Pdims.tau_bar*(Pdims.J-2))

    # Compute the block projection
    proj_block = zeros(size(P_OLS)[1])
    for k in 1:Pdims.J*Pdims.tau_bar*(Pdims.J-2)
        proj_block = proj_block + P_OLS*v_blocks[:,k]
    end
    proj_block = (1/(Pdims.J*Pdims.tau_bar*(Pdims.J-2)))*proj_block

    # Stack up to a vector
    return proj_block
end

function project_to_matrix(M::AbstractArray{T},P_mat::Matrix{Float64},Pdims) where T
    M_blocks = reshape(M,(Pdims.J-1)*(Pdims.T-1),Pdims.J*Pdims.tau_bar*(Pdims.J-2),size(M)[2])

    # Compute the block projection
    proj_block = zeros((Pdims.J-1)*(Pdims.T-1),size(M)[2])
    for k in 1:Pdims.J*Pdims.tau_bar*(Pdims.J-2)
        proj_block = proj_block + P_mat*M_blocks[:,k,:]
    end
    proj_block = 1/(Pdims.J*Pdims.tau_bar*(Pdims.J-2))*proj_block

    # Stack up to a matrix
    proj_block_stacked = repeat(proj_block, outer=Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    return proj_block_stacked
end

function project_orthogonal_complement(v::AbstractVector{T},P_mat::Matrix{Float64},Pdims) where T
    # Get the projection on the orthogonal complement
    return v - project(v,P_mat,Pdims)
end

function project_orthogonal_complement_matrix(M::AbstractArray{T},P_mat::Matrix{Float64},Pdims) where T
    # Get the projection on the orthogonal complement
    return M - project_to_matrix(M,P_mat,Pdims)
end

function inv_resid(θ::AbstractVector{T},P_mat::Matrix{Float64},D,Pdims) where T
    # Read data
    Y = D.Y
    X = D.X
    Z = D.Z

    # Cmpute the residual
    Y_hat = D.X*θ
    resid = Y - Y_hat

    # Project the residual on the orthogonal complement of the dummy matrix
    if Pdims.FEs == true
        resid_projected = project_orthogonal_complement(resid,P_mat,Pdims)
        resid_projected = resid_projected[Int.(D.indices_observed)]
        return resid_projected
    else
        resid = resid[Int.(D.indices_observed)]
        return resid
    end
end

function emp_moments(theta::AbstractVector{T},P_mat::Matrix{Float64},D,Pdims::NamedTuple) where T
    resid = inv_resid(theta,P_mat,D,Pdims)
    if Pdims.FEs == true
        Z_projected = project_orthogonal_complement_matrix(D.Z,P_mat,Pdims)
        Z_projected = Z_projected[Int.(D.indices_observed),:]
        e_mom = Z_projected .* resid
    else
        Z = D.Z[Int.(D.indices_observed),:]
        e_mom = Z .* resid
    end
    return e_mom
end

function g_hat(theta::AbstractVector{T},P_mat::Matrix{Float64},D,Pdims::NamedTuple) where T
    mom_mat = emp_moments(theta,P_mat,D,Pdims)
    N = length(D.indices_observed)
    g_h = (1/N)*vec(sum(mom_mat,dims=1)) # Sx1 vector
    return g_h
end

function gmm(theta::AbstractVector{T},W::Matrix{Float64},P_mat::Matrix{Float64},D,Pdims::NamedTuple) where T
    g_h = g_hat(theta,P_mat,D,Pdims)
    gmm_objective = g_h'*W*g_h
    return gmm_objective
end

function omega_mat(theta::AbstractVector{T},P_mat::Matrix{Float64},D,Pdims::NamedTuple) where T
    mom_mat = emp_moments(theta,P_mat,D,Pdims)'
    g_h = g_hat(theta,P_mat,D,Pdims)
    mom_mat_decentralized = mom_mat .- g_h
    N = length(D.indices_observed)
    om_mat = (mom_mat_decentralized*mom_mat_decentralized')/N
    return om_mat
end

function lambda_projected_logs(theta::AbstractVector{T},P_mat::Matrix{Float64},D,Pdims) where T
    Y = D.Y
    X = D.X
    Z = D.Z

    # Compute residuals
    Y_hat = D.X*theta
    Y_aux = Y - Y_hat #J*T vector

    lambda_hat = ols_mat(Y_aux,P_mat,Pdims)
    return lambda_hat
end

function gmm_SE(theta::AbstractVector,W::Matrix{Float64},P_mat::Matrix{Float64},D,Pdims::NamedTuple)
    g_hat_function(x) =  g_hat(x,P_mat,D,Pdims)
    G_hat(x) = ForwardDiff.jacobian(g_hat_function, x)
    G = G_hat(theta)
    Σ = inv(G'*W*G)/((Pdims.J-1)*(Pdims.T-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    SE = sqrt.(diag(Σ))
    return SE
end