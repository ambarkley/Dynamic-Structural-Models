using StatsKit, ForwardDiff, Ipopt, NLsolve, Optim, Parameters, Zygote, LinearAlgebra, Random, Plots, BenchmarkTools

"""
Example for the EM algorithm using a finite Poisson mixture
"""

function em_algo_example()
    Random.seed!(19);
    nobs=1000;
    mix_dist = MixtureModel(Poisson[Poisson(2.0), Poisson(8.0)], [0.4, 0.6]);
    Y=rand(mix_dist,nobs);

    ex_hist = histogram(Y,normalize=:pdf, bins=0:12);

    π=[0.5, 0.5];
    θ=[1.0, 2.0];
    iter=1;
    maxiter=1000;
    lik0=10.0;
    likdiff=10.0;
    tol=1e-6;
    while likdiff>tol && iter<=maxiter
        EZ = [(π[j]*pdf(Poisson(θ[j]),Y[i]))/(sum(π[k]*pdf(Poisson(θ[k]),Y[i]) for k∈eachindex(π))) for i ∈ eachindex(Y), j ∈ eachindex(π)];
        theta_fun(θ) = -sum(EZ[i,j]*log(pdf(Poisson(θ[j]),Y[i])) for i ∈ eachindex(Y), j ∈ eachindex(π))
        π=mean(EZ,dims=1);
        b=optimize(theta_fun,θ);
        # Calculate likelihood and update for next iteration:
        θ=b.minimizer;
        lik1 = -theta_fun(θ) + sum(log(π[j])*EZ[i,j] for i∈eachindex(Y), j∈eachindex(π));
        likdiff=abs(lik1-lik0);
        lik0=lik1;
        iter=iter+1;
    end

    return θ,π,iter, ex_hist
end

"""
CCP estimation of the Rust (1987) bus engine replacement model with unobserved heterogeneity
"""

"""
These two functions perform value function iteration and generate the data
"""

function value_function_iteration(X::AbstractRange{Float64},S::Vector{Int64},F1::Matrix{Float64},F2::Matrix{Float64},β::Number,θ::Vector;MaxIter=1000)
    x_len=length(X);
    γ=0.5772;
    value_function2=zeros(x_len,length(S));
    value_diff=1.0;
    tol=1e-5;
    iter=1;
    local v1, v2
    while (value_diff>tol) && (iter<=MaxIter)
        value_function1=value_function2;
        v1=[0.0 + β*F1[j,:]'*value_function1[:,s] for j∈eachindex(X), s∈eachindex(S)];
        v2=[θ[1]+θ[2]*X[j]+θ[3]*S[s] + β*(F2[j,:]'*value_function1[:,s]) for j=1:x_len, s∈eachindex(S)];
        value_function2=[log(exp(v1[j,s])+exp(v2[j,s]))+γ for j=1:x_len, s=1:length(S)];
        iter=iter+1;
        #value_diff=sum((value_function1 .- value_function2).^2);
        value_diff=maximum((value_function1 .- value_function2).^2);
    end
    ccps=[1/(1+exp(v2[j,s,]-v1[j,s])) for j=1:x_len, s=1:length(S)];
    return (ccps_true=ccps, value_function=value_function2)
end

function generate_data(N,T,X,S,F1,F2,F_cumul,β,θ;T_init=10,π=0.4,ex_initial=0)
    if ex_initial==1
        T_init=0;
    end
    x_data=zeros(N,T+T_init);
    x_data_index=Array{Int32}(ones(N,T+T_init));
    if ex_initial==1
        x_data_index[:,1]=rand(1:length(X),N,1);
        x_data[:,1]=X[x_data_index[:,1]];
    end
    s_data=(rand(N) .> π) .+ 1;
    d_data=zeros(N,T+T_init);

    draw_ccp=rand(N,T+T_init);
    draw_x=rand(N,T+T_init);

    (ccps,_)=value_function_iteration(X,S,F1,F2,β,θ);

    for n=1:N
        for t=1:T+T_init
            d_data[n,t]=(draw_ccp[n,t] > ccps[x_data_index[n,t],s_data[n]])+1;
            if t<T+T_init
                x_data_index[n,t+1]=1 + (d_data[n,t]==2)*sum(draw_x[n,t] .> F_cumul[x_data_index[n,t],:]); 
                x_data[n,t+1]=X[x_data_index[n,t+1]];
            end
        end
    end

    return (XData=x_data[:,T_init+1:T+T_init], SData=repeat(s_data,1,T),
        DData=d_data[:,T_init+1:T+T_init],
        XIndexData=x_data_index[:,T_init+1:T_init+T],
        TData=repeat(1:T,N,1),
        NData=repeat((1:N)',1,T)) 
end

"""
Estimating the dynamic model parameters using the EM algorithm
"""


function ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β)
    γ=0.5772;
    x_len=length(X);
    v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
    v2_ccp=[θ[1]+θ[2]*X[j] + θ[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)];
    q_use=repeat(q[:,1],1,T);
    ccp_lik_1=-sum(q_use[n,t]*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) - log(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]))) for n=1:N, t=1:T);
    ccp_lik_2=-sum((1-q_use[n,t])*((DData[n,t]==2.0)*(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) - log(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]))) for n=1:N, t=1:T);
    return ccp_lik_1 + ccp_lik_2
end


function unobserved_het_estimation()
    x_min=0.0;
    x_max=10.0;
    x_int=0.5;
    x_len=Int32(1+(x_max-x_min)/x_int);
    x=range(x_min,x_max,x_len);

    # Transition matrix for mileage:
    x_tran       = zeros((x_len, x_len));
    x_tran_cumul = zeros((x_len, x_len));
    x_tday      = repeat(x, 1, x_len); 
    x_next      = x_tday';
    x_zero      = zeros((x_len, x_len));

    x_tran = (x_next.>=x_tday) .* exp.(-(x_next - x_tday)) .* (1 .- exp(-x_int));
    x_tran[:,end]=1 .-sum(x_tran[:,1:(end-1)],dims=2);
    x_tran_cumul=cumsum(x_tran,dims=2);

    S=[1, 2];
    s_len=Int32(length(S));
    F1=zeros(x_len,x_len);
    F1[:,1].=1.0;
    F2=x_tran;

    N=1000;
    T=40;
    X=x;
    θ=[2.0, -0.15, 1.0];
    β=0.9;
    F_cumul=x_tran_cumul;
    Random.seed!(17);
    XData, SData, DData, XIndexData, TData, NData = generate_data(N,T,X,S,F1,F2,F_cumul,β,θ; ex_initial=1);
    γ=0.5772;

    π=0.5;
    q=[π*ones(N,1) (1-π)*ones(N,1)];
    ccp_hat = [sum(repeat(q[:,s],T,1).*(XIndexData[:] .== j).* (DData[:] .== 1.0))/sum(repeat(q[:,s],T,1).*(XIndexData[:] .== j)) for j∈eachindex(X), s∈eachindex(S)];
    
    iter=1;
    cond=0;
    max_iter=1000;
    tol=1e-6;
    lik0=1.0;
    theta_0=[0.1,0.1,0.1];
    stored_lik_vals=zeros(max_iter,1);
    while cond==0 && iter<=max_iter
        v1_ccp=repeat(β*(-log.(ccp_hat[1,:]') .+ γ),x_len,1);
        v2_ccp=[theta_0[1]+theta_0[2]*X[j] + theta_0[3]*S[s] + β*(F2[j,:]'*(-log.(ccp_hat[:,s])) +γ) for j∈eachindex(X), s∈eachindex(S)]; # Check matrix multiplication compatability

        # Pointwise likelihood:

        like_pointwise_1= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],1]-v1_ccp[XIndexData[n,t],1])) for n=1:N, t=1:T];
        like_pointwise_2= [((DData[n,t]==2.0)*exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2]) +(1-(DData[n,t]==2.0)))/(1+exp(v2_ccp[XIndexData[n,t],2]-v1_ccp[XIndexData[n,t],2])) for n=1:N, t=1:T];
        ll_pw1=prod(like_pointwise_1,dims=2);
        ll_pw2=prod(like_pointwise_2,dims=2);

        q[:,1] .= (π.*ll_pw1)./(π.*ll_pw1 .+ (1-π).*ll_pw2);
        q[:,2] .= 1.0 .- q[:,1];
        π=mean(q[:,1]);
        # Maximization
        ccp_hat = [sum(repeat(q[:,s],T,1).*(XIndexData[:] .== j).* (DData[:] .== 1.0))/sum(repeat(q[:,s],T,1).*(XIndexData[:] .== j)) for j∈eachindex(X), s∈eachindex(S)];
        inner_lik(θ) = ccp_likelihood_inner(θ,N,T,X,S,F2,XIndexData,DData,q,ccp_hat,β);
        optim_res = optimize(inner_lik,theta_0,LBFGS(); autodiff = :forward);
        theta_0=optim_res.minimizer;

        stored_lik_vals[iter]=sum(q[:,1].*log.(ll_pw1) + q[:,2].*log.(ll_pw2)) + sum(log(π)*q[:,1] + log(1-π)*q[:,2]);
        #lik_diff = abs(lik0-lik1);
        #lik0=lik1;
        if iter>25 
            lik_diff=abs((stored_lik_vals[iter]-stored_lik_vals[iter-25])/stored_lik_vals[iter-25]);
            if lik_diff<tol
                cond=1;
            end
        end
        iter=iter+1;
        if iter>max_iter
            print("Maximum number of iterations reached")
            break
        end
    end

    return theta_0, π, iter
end

θ_hat, π_hat, num_iter = unobserved_het_estimation();

"""
Display results
"""
θ_1, θ_2, θ_3 =[2.0, -0.15, 1.0];
theta_hat_1, theta_hat_2, theta_hat_3 = round.(θ_hat, digits = 4);
π = 0.4;
println("----------------------------")
println("CCP estimation with unobserved heterogeneity results: \n ----------------------------")
println("True θ_1: $θ_1 \nEstimated θ_1: $theta_hat_1\n")
println("True θ_2: $θ_2 \nEstimated θ_2: $theta_hat_2 \n")
println("True θ_3: $θ_3 \nEstimated θ_3: $theta_hat_3 \n")
println("True π: $π \nEstimated θ_3: $π_hat\n")