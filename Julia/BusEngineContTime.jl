#------------------------------------------
# Continuous-time implementation of dynamic discrete choice estimation
# Code follows Blevins (2024)

# Prepared by Aaron Barkley, University of Melbourne
# Last updated: August 2025
#------------------------------------------

using StatsKit, ForwardDiff, Ipopt, NLsolve, Optim, Parameters, Zygote, LinearAlgebra, Random, Plots, BenchmarkTools

struct FixedParameters{TI<:Integer, TF<:AbstractFloat}
    λ::TF
    ρ::TF
    q::TF
    nStates::TI
    maxMileage::TI
    mileStates::AbstractVector{TF}
end

function FixedParameters(;  λ::TF=1.0,
                            ρ::TF = 0.05,
                            q::TF=2.0,
                            nStates::TI =50,
                            maxMileage::TI = 10
                            ) where {TI<:Integer, TF<:AbstractFloat}
        mileStates = range(0.0,maxMileage, length = nStates)
    return FixedParameters(λ, ρ, q, nStates, maxMileage, mileStates)
end

fix_par = FixedParameters()
nStates = fix_par.nStates
F1 = [j == i+1 ? 1.0 : 0.0 for i = 1:nStates, j = 1:nStates]
F1[end,end] = 1.0

F0 = hcat(ones(nStates,1), zeros(nStates, nStates-1))
θ = [-3.0; -14.0]

function value_function_iteration(θ, fix_par; MaxIter=1000)
    x_len=fix_par.nStates;
    ρ = fix_par.ρ
    q = fix_par.q
    λ = fix_par.λ

    mileStates = fix_par.mileStates
    γ=0.5772;
    V_new=zeros(x_len,1);
    value_diff=1.0;
    tol=1e-5;
    iter=1;
    #local v1, v2
    while (value_diff>tol) && (iter<=MaxIter)
        V=V_new;

        V_new = [(1.0/(ρ + q + λ))*(θ[1]*mileStates[j] + q*V[min(j+1,x_len)] + λ*(log(exp(θ[2] + V[1]) + exp(V[j])) + γ)) for j ∈ 1:x_len]
        iter=iter+1;
        value_diff=maximum((V .- V_new).^2);
    end
    ccps=[1/(1+exp((θ[2] + V_new[1]) - V_new[j])) for j=1:x_len];
    return  (true_ccps = ccps, V = V_new)
end

ccps, val_fun = value_function_iteration(θ, fix_par)


function generate_data(T, ccps_true, fix_par)
    mileStates = fix_par.mileStates
    maxMileage = fix_par.maxMileage
    nStates = fix_par.nStates
    # Draw exponential random variables to simulate data
    t = 0.0
    t_data = [];
    x_index = [1];
    X = [mileStates[1]];
    event_list = [];
    while t < T
        # draw rates for moves
        q_draw, lambda_draw = randexp(1,2)
        
        if q_draw < lambda_draw
            # update mileage
            push!(X, mileStates[min(x_index[end]+1, nStates)])
            push!(t_data, t + q_draw)
            push!(x_index, min(nStates, x_index[end] + 1))
            push!(event_list, 0)
            t += q_draw
        else
            ccp_draw = rand()
            if ccp_draw > ccps_true[x_index[end]]
                push!(x_index, 1)
                push!(X, mileStates[1])
                push!(t_data, t+lambda_draw)
                push!(event_list, 1)
            end
            t += lambda_draw
        end
    end

    t_data = pushfirst!(t_data, 0.0)
    return (X=X, XIndex=x_index,
        TData=t_data, EventList = event_list) 
end


T = 1000.0

X, x_index, t_data, events = generate_data(T, ccps, fix_par);

t_elapsed = [t_data[j+1] - t_data[j] for j=1:(length(t_data) - 1)]
x_index = x_index[1:end-1]
X = X[1:end-1]

function fiml_likelihood(θ, x_index, t_elapsed, events, fix_par)
    ccps, _ = value_function_iteration(θ, fix_par)
    λ = fix_par.λ
    q = fix_par.q
    # likelihood in continuous time
    numerator = (events .== 0).*q + (events .== 1) .*λ.*((1.0 .- ccps[x_index]))
    denominator = exp.(- t_elapsed.*(q .+ λ.*(1.0 .- ccps[x_index])))

    return -sum(log.(numerator.*denominator))

end

function fiml_estimation(x_index, t_elapsed, events, fix_par)
    
    f(θ) = fiml_likelihood(θ, x_index, t_elapsed, events, fix_par)
    result = optimize(f,[-1.0; -1.0], LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward);
    #result = optimize(f,[0.1;0.1;0.1]);
    return result.minimizer
end

function ccp_est_lik(α, x_index, X, t_elapsed, events, fix_par)
    q = fix_par.q
    λ = fix_par.λ
    maxMileage = fix_par.maxMileage
    
    X_logit = hcat(ones(size(x_index,1)), X, (X.^2)./10.0, (X.>(maxMileage/1.5)))
    ccp_logit = exp.(X_logit*α)./(1.0 .+ exp.(X_logit*α))

    numerator = (events .== 0).*q + (events .== 1) .*λ.*((1.0 .- ccp_logit[x_index]))
    denominator = exp.(- t_elapsed.*(q .+ λ.*(1.0 .- ccp_logit[x_index])))
    return -sum(log.(numerator.*denominator))
end

function stationary_ccp_main(x_index, X, F1, t_elapsed, events, fix_par)
    nStates = fix_par.nStates
    mileStates = fix_par.mileStates
    maxMileage = fix_par.maxMileage
    alpha_hat = optimize(α->ccp_est_lik(α, x_index, X, t_elapsed, events, fix_par), [3.0; 1.0; -2.0; 1.0], LBFGS(); autodiff = :forward).minimizer
    X_logit = hcat(ones(nStates), mileStates, (mileStates.^2)./10.0, (mileStates.>(maxMileage/1.5)))
    ccp_fit = exp.(X_logit*alpha_hat)./(1.0 .+ exp.(X_logit*alpha_hat)) 
    #θ = [-3.0; -14.0]
    #ccps, _ = value_function_iteration(θ, fix_par)
    #ccp_fit = ccps
    theta_0 = [-2.0; -10.0]
    ccp_fun(θ) = ccp_struct_lik(θ, ccp_fit, events, t_elapsed, x_index, F1, fix_par)
    result = optimize(ccp_fun, theta_0, LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward)
    return result.minimizer
end

function ccp_struct_lik(θ, ccp_fit, events, t_elapsed, x_index, F1, fix_par)
    nStates = fix_par.nStates
    mileStates = fix_par.mileStates
    λ = fix_par.λ
    ρ = fix_par.ρ
    q = fix_par.q
    γ = 0.5772
    Q_0 = [-q*(i == j) + q*(i+1 == j) for i = 1:nStates, j = 1:nStates]
    Q_0[end, end] = 0.0;
    u = θ[1].*mileStates;
    c = θ[2]
    
    Σ = [ccp_fit[i]*(i==j) + (1.0 - ccp_fit[i])*(j==1) for i ∈ 1:nStates, j ∈ 1:nStates]
    V_ccp = ((ρ .+ λ)*Matrix(I,nStates,nStates) .- λ*Σ .- Q_0 )\(u .+ λ*(ccp_fit.*(0.0 .+ γ .- log.(ccp_fit)) .+ (1.0 .- ccp_fit).*(c.+ γ .- log.(1.0 .- ccp_fit))))
    
    model_ccp = [1.0/(1.0 + exp((θ[2] + V_ccp[1]) - V_ccp[j])) for j=1:nStates];
    numerator = (events .== 0).*q .+ (events .== 1) .*λ.*((1.0 .- model_ccp[x_index]))
    denominator = exp.(- t_elapsed.*(q .+ λ.*(1.0 .- model_ccp[x_index])))

    return -sum(log.(numerator.*denominator))
end

function monte_carlo(Nsim,θ, fix_par; T = 1000.0)
    theta_est = zeros(Nsim,2)
    Random.seed!(2023)
    for n=1:Nsim
        #T = 1000.0
        ccps, val_fun = value_function_iteration(θ, fix_par)
        X, x_index, t_data, events = generate_data(T, ccps, fix_par);

        t_elapsed = [t_data[j+1] - t_data[j] for j=1:(length(t_data) - 1)]
        x_index = x_index[1:end-1]

        theta_est[n,:] .= fiml_estimation(x_index, t_elapsed, events, fix_par)
    end

    return theta_est
end

"""
FIML estimation Monte Carlo
"""
theta_mc = monte_carlo(200, θ, fix_par)

function ccp_monte_carlo(Nsim,θ, fix_par; T = 1000.0)
    theta_est = zeros(Nsim,2)
    Random.seed!(2024)
    for n=1:Nsim
        #T = 1000.0
        ccps, val_fun = value_function_iteration(θ, fix_par)
        X, x_index, t_data, events = generate_data(T, ccps, fix_par);

        t_elapsed = [t_data[j+1] - t_data[j] for j=1:(length(t_data) - 1)]
        x_index = x_index[1:end-1]
        X = X[1:end-1]

        theta_est[n,:] .= stationary_ccp_main(x_index, X, F1, t_elapsed, events, fix_par)
    end

    return theta_est
end

"""
NOTE: the CCP Monte Carlo will give slightly inaccurate estimates for θ due to first-stage CCP bias
(one can verify this by using the true ccps in the estimation by uncommenting a few lines)

A more sophisticated strategy than the simple logit used above
for first-stage CCP estimation will yield better results.
"""

theta_mc_ccp = ccp_monte_carlo(200, θ, fix_par; T = 10000.0); # Simulates/estimates 200 times
