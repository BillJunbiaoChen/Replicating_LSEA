#################################################################################################################
# HELPER FUNCTIONS
#################################################################################################################


function predict_levels(θ::AbstractVector{T},data,Pdims, γ) where T
    # Read parameters
    aux_alpha = θ[1:Pdims.K*Pdims.S];
    λ = θ[(Pdims.K*Pdims.S+1):end];

    # Read data
    X = data.X;
    N = size(X)[1];
    D = data.D;
    A = data.A;

    y_hat = zeros(T,N);
    total_budget = X*(aux_alpha);

    if time_FE + location_FE  > 0
            y_hat = total_budget.*exp.(D*[0;λ]).*A.^(γ)
    else
            y_hat = total_budget.*A.^(γ)
    end

    #
    #

    return y_hat
end

function predict_logs(θ::AbstractVector{T},data,Pdims, γ) where T
    # Read parameters

    aux_alpha = θ[1:Pdims.K*Pdims.S];
    λ = θ[(Pdims.K*Pdims.S+1):end];


    # Read data
    X = data.X;
    N = size(X)[1];
    D = data.D;
    A = data.A;

    total_budget = X*(aux_alpha);
    y_log_hat = zeros(T,N);

    if time_FE + location_FE  > 0
            y_log_hat = D*[0;λ]+γ*log.(A)+log.(total_budget)
    else
            y_log_hat = γ*log.(A)+log.(total_budget)
    end

    return y_log_hat
end

function resid_logs(θ::AbstractVector{T},data,Pdims, γ) where T

    log_y_hat = predict_logs(θ,data,Pdims, γ)

    Y = data.Y;

    resid = zeros(T,N);
    resid = log.(Y) - log_y_hat;

    return resid
end



function gmm_logs_bayesian(θ::AbstractVector{T},W,data,Pdims,weight, γ) where T

    Y = data.Y;
    log_y_hat = predict_logs(θ,data,Pdims, γ)
    resid = zeros(T,N);
    resid = log.(Y) - log_y_hat;
    Z_full = data.Z_full;
    mom = Z_full.*resid;
    g = zeros(T,size(mom)[2]);
    g = (1/N)*sum(weight.*mom, dims=1)';
    gmm_objective = sum(g'*W*g)

    return gmm_objective
end

function gmm_logs(θ::AbstractVector{T},W,data,Pdims, γ) where T

    Y = data.Y;
    log_y_hat = predict_logs(θ,data,Pdims, γ)
    resid = zeros(T,N);
    resid = log.(Y) - log_y_hat;
    Z_full = data.Z_full;
    mom = Z_full.*resid;
    g = zeros(T,size(mom)[2]);
    g = (1/N)*sum(mom, dims=1)';
    gmm_objective = sum(g'*W*g)

    return gmm_objective
end