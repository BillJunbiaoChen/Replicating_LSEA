#################################################################################################################
# HELPER FUNCTIONS
#################################################################################################################


"""
function reshape_mnl_to_model(M::Matrix{Float64},Pdims::NamedTuple)
    Prepares provided CCPs into a custom format for further analysis
"""
function reshape_mnl_to_model(M::Matrix{Float64},Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.T,Pdims.J*Pdims.tau_bar,Pdims.J)
    output_tensor = zeros(Pdims.T,Pdims.J,Pdims.J*Pdims.tau_bar)
    for t in 1:Pdims.T
        output_tensor[t,:,:] = M_tensor[t,:,:]'
    end
    output_matrix = reshape(output_tensor,Pdims.T*Pdims.J,Pdims.J*Pdims.tau_bar)
    return output_matrix
end

"""
function normalize_mat(M::Matrix{Float64},Pdims::NamedTuple)
    Normalizes a given matrix with respect to the entries corresponding to the outside option.
"""
function normalize_mat(M::Matrix{Float64},Pdims::NamedTuple)
    v_outside_mat = M[(Pdims.J-1)*(Pdims.T)+1:end,:]
    v_outside_mat_stacked = kron(ones(Pdims.J),v_outside_mat)
    v_mat_norm = M - v_outside_mat_stacked
    return v_mat_norm
end

"""
function Y_next_j(j::Int64,log_CCP::Matrix{Float64},Pdims::NamedTuple)
    For a given location j, compute the term in Y pertaining to the transition from j to renewal actions.
"""
function Y_next_j(j::Int64,log_CCP::Matrix{Float64},Pdims::NamedTuple)
    @unpack T, J, tau_bar = Pdims
    # Prepare results
    Y_tnext_j = zeros(Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T)

    log_CCP_tensor = reshape(copy(log_CCP),Pdims.T,Pdims.J,Pdims.tau_bar,Pdims.J)

    # Exclude 0 and j as the renewal actions, and shift the time periods one period backwards
    log_CCP_tensor_reduced_j = log_CCP_tensor
    log_CCP_tensor_reduced_j[1:end-1,:,:,:] = log_CCP_tensor_reduced_j[2:end,:,:,:]
    log_CCP_tensor_reduced_j[end,:,:,:] = zeros(Pdims.J,Pdims.tau_bar,Pdims.J)

    # Compose all Y's for j
    renewal_actions_j = [d for d in 1:Pdims.J if d!=j && d!= Pdims.J]

    for t in 1:Pdims.T
        for j_prev in 1:Pdims.J
            for tau_prev in 1:Pdims.tau_bar
                for (id_renewal,renewal_action) in enumerate(renewal_actions_j)
                    prob_dist_tau_j = Pdims.Gamma_tensor[tau_prev,j_prev,t,j,:]
                    prob_dist_tau_0 = Pdims.Gamma_tensor[tau_prev,j_prev,t,Pdims.J,:]
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

"""
function Y_next(log_CCP::Matrix{Float64},Pdims::NamedTuple)
    Combines the term in Y pertaining to the transition from j to renewal actions for each j into a single matrix.
"""
function Y_next(log_CCP::Matrix{Float64},Pdims::NamedTuple)
    Y_next_j_vecs = [Y_next_j(j,copy(log_CCP),Pdims) for j in 1:Pdims.J-1]
    return vcat(Y_next_j_vecs...)
end

"""
function dist_est_j(j::Int64,dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    Prepares the moving cost portion for location j
"""
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
end

"""
function dist_est(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    Combines the moving cost portions for all locations.
"""
function dist_est(dist_mat::Matrix{Float64},Pdims::NamedTuple,Pparam::NamedTuple)
    dist_est_vecs = [dist_est_j(j,dist_mat,Pdims,Pparam) for j in 1:Pdims.J-1]
    return vcat(dist_est_vecs...)
end

"""
function remove_T_vec(v::Vector{Float64},Pdims::NamedTuple)
    Removes all entries for the las period T from a specified vector.
"""
function remove_T_vec(v::Vector{Float64},Pdims::NamedTuple)
    v_mat = reshape(v,(Pdims.J-2)*Pdims.J*Pdims.tau_bar,Pdims.T,Pdims.J-1)
    v_mat_reduced = v_mat[:,1:end-1,:]
    v_reduced = reshape(v_mat_reduced,(Pdims.T-1)*(Pdims.J-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    return v_reduced
end

"""
function remove_T_mat(M::AbstractArray,Pdims::NamedTuple)
    Removes all entries for the las period T from a specified matrix.
"""
function remove_T_mat(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,(Pdims.J-2)*Pdims.J*Pdims.tau_bar,Pdims.T,Pdims.J-1,size(M)[2])
    M_tensor_reduced = M_tensor[:,1:end-1,:,:]
    M_reduced = reshape(M_tensor_reduced,(Pdims.T-1)*(Pdims.J-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2),size(M)[2])
    return M_reduced
end

"""
function invert_indices_matrix(M::AbstractArray,Pdims::NamedTuple)
    Restacks a given matrix with rows indexed by (j,t,j_0,tau_0,renewal) to (j_0,tau_0,j,t,renewal)
"""
function invert_indices_matrix(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T-1,Pdims.J-1,size(M)[2])
    inverted_M_tensor = zeros(Pdims.T-1,Pdims.J-2,Pdims.J-1,Pdims.tau_bar,Pdims.J,size(M)[2])
    for j_init in 1:Pdims.J
        for tau_init in 1:Pdims.tau_bar
            for j_renewal in 1:Pdims.J-2
                inverted_M_tensor[:,j_renewal,:,tau_init,j_init,:] = M_tensor[j_renewal,tau_init,j_init,:,:,:]
            end
        end
    end
    inverted_M_mat = reshape(inverted_M_tensor,(Pdims.T-1)*(Pdims.J-1)*(Pdims.J-2)*Pdims.tau_bar*Pdims.J,size(M)[2])
    return inverted_M_mat
end

"""
function invert_indices_vector(M::AbstractArray,Pdims::NamedTuple)
    Restacks a given vector with rows indexed by (j,t,j_0,tau_0,renewal) to (j_0,tau_0,j,t,renewal)
"""
function invert_indices_vector(M::AbstractArray,Pdims::NamedTuple)
    M_tensor = reshape(M,Pdims.J-2,Pdims.tau_bar,Pdims.J,Pdims.T-1,Pdims.J-1)
    inverted_M_tensor = zeros(Pdims.T-1,Pdims.J-2,Pdims.J-1,Pdims.tau_bar,Pdims.J)
    for j_init in 1:Pdims.J
        for tau_init in 1:Pdims.tau_bar
            for j_renewal in 1:Pdims.J-2
                inverted_M_tensor[:,j_renewal,:,tau_init,j_init] = M_tensor[j_renewal,tau_init,j_init,:,:]
            end
        end
    end
    inverted_M_mat = reshape(inverted_M_tensor,(Pdims.T-1)*(Pdims.J-1)*(Pdims.J-2)*Pdims.tau_bar*Pdims.J)
    return inverted_M_mat
end

"""
function Γ_tensor(Pdims::NamedTuple,tau_trans_probs::DataFrame)
    Prepare a tensor of transition probabilities for location capital
"""
function Γ_tensor(Pdims::NamedTuple,tau_trans_probs::DataFrame)
    Gamma_tensor = zeros(Float64,Pdims.tau_bar,Pdims.J,Pdims.T,Pdims.J,Pdims.tau_bar)
    for j_prime in 1:Pdims.J
        for t in 1:Pdims.T
            for j in 1:Pdims.J
                if j == j_prime
                    trans_prob_12 = tau_trans_probs[(tau_trans_probs.gb .== j) .& (tau_trans_probs.year .== Pdims.lower_bound_year_sample - 1 + t),"transition_prob"][1]
                    Gamma_tensor[:,j,t,j_prime,:] = [(1-trans_prob_12) trans_prob_12; 0 1]
                else
                    Gamma_tensor[:,j,t,j_prime,1] = ones(Pdims.tau_bar)
                end
            end
        end
    end
    return Gamma_tensor
end


#################################################################################################################
# MAIN FUNCTION
#################################################################################################################


"""
function prepare_invariant_data(CCP_df::DataFrame, dist_mat::Matrix{Float64}, Pdims::NamedTuple, Pparam::NamedTuple, g::Int64)
    Prepares ECCP paths for invariant variables.

    Inputs:
        - CCP_df::DataFrame, should contain columns p_gb p_tau year t and phatk for k indexing locations
        - dist_mat::Matrix{Float64}, distance matrix
        - Pdims::NamedTuple, a named tuple with dimensions
        - Pparam::NamedTuple, a named tuple parameter values
        - g::Int64, type index

    Outputs:
        - a dataframe with prepared data
"""
function prepare_invariant_data(CCP_df::DataFrame, dist_mat::Matrix{Float64}, Pdims::NamedTuple, Pparam::NamedTuple, g::Int64)

    #### Compute Y
    # Resort the dataframe
    sort!(CCP_df,[:p_gb,:p_tau,:t])
    CCP = Matrix{Float64}(CCP_df[!,r"phat"])

    # Compose CCPs
    CCP = reshape_mnl_to_model(CCP,Pdims)
    
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
    Gamma_mat_all = reshape(Pdims.Gamma_tensor,Pdims.J*Pdims.tau_bar*Pdims.T*Pdims.J,Pdims.tau_bar)
    Gamma_mat_avg = Gamma_mat_all*[0,1]
    Gamma_mat = reshape(Gamma_mat_avg,Pdims.J*Pdims.tau_bar*Pdims.T,Pdims.J)

    # Normalize Gamma_mat by columns
    Gamma_mat_norm = Gamma_mat .- Gamma_mat[:,Pdims.J]

    # Reshape to stacked vectors after removing the outside option
    Gamma_mat_norm_noout = Gamma_mat_norm[:,1:Pdims.J-1]
    Gamma_mat_norm_noout_T = reshape(Gamma_mat_norm_noout,Pdims.J*Pdims.tau_bar,(Pdims.J-1)*Pdims.T)
    Gamma_mat_norm_noout_vec = reshape(Gamma_mat_norm_noout_T,(Pdims.J-1)*Pdims.J*Pdims.tau_bar*Pdims.T)

    # Repeat J-2 times to since this is the same for all renewal actions
    tau_vec_full = repeat(Gamma_mat_norm_noout_vec,inner=Pdims.J-2)

    #### Compose variables
    X = [dist_mat_full tau_vec_full]

    #### Remove the last time period as we are not able to identify it
    Y_reduced = remove_T_vec(Y,Pdims)
    X_reduced = remove_T_mat(X,Pdims)

    #### Restack the matrices
    Y_reduced_inverted = invert_indices_vector(Y_reduced,Pdims)
    X_reduced_inverted = invert_indices_matrix(X_reduced,Pdims)

    # Put into a dataframe and put the indices back
    df_X = DataFrame(X_reduced_inverted,["gamma_0_vec","gamma_1_vec","gamma_2_vec","tau_vec_full"])
    df_Y = DataFrame(reshape(Y_reduced_inverted,length(Y_reduced_inverted),1),["y"])
    df_dat = hcat(df_X,df_Y)
    df_vars = DataFrame()
    indices_j_t = collect(1:Pdims.J-1)
    indices_j_t = repeat(indices_j_t, inner = (Pdims.T-1)*(Pdims.J-2))
    indices_j_t = repeat(indices_j_t, outer = Pdims.J*Pdims.tau_bar)
    indices_t = collect(1:Pdims.T-1)
    indices_t = repeat(indices_t, outer = (Pdims.J-1)*Pdims.J*Pdims.tau_bar*(Pdims.J-2))
    indices_j_t_1 = collect(1:Pdims.J)
    indices_j_t_1 = repeat(indices_j_t_1,inner = (Pdims.J-2)*Pdims.tau_bar*(Pdims.J-1)*(Pdims.T-1))
    indices_tau_t_1 = collect(1:Pdims.tau_bar)
    indices_tau_t_1 = repeat(indices_tau_t_1, inner = (Pdims.J-2)*(Pdims.J-1)*(Pdims.T-1))
    indices_tau_t_1 = repeat(indices_tau_t_1, outer = Pdims.J)
    renewals = vcat([[j_r for j_r in 1:Pdims.J-1 if j_r != j] for j in 1:Pdims.J-1]...)
    indices_renewal = repeat(renewals,inner = Pdims.T-1)
    indices_renewal = repeat(indices_renewal, outer = Pdims.J*Pdims.tau_bar)
    df_vars[!,"gb"] = indices_j_t
    df_vars[!,"p_gb"] = indices_j_t_1
    df_vars[!,"period"] = indices_t
    df_vars[!,"p_tau"] = indices_tau_t_1
    df_vars[!,"renewal"] = indices_renewal
    df_vars[!,"combined_cluster"] .= g
    df_all = hcat(df_dat,df_vars)

    return df_all
end