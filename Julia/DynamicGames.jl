using Random, LinearAlgebra, Statistics, DataFrames, GLM
using Optim, ForwardDiff, StatsFuns, Plots

# ----------------------------
# Utility helpers
# ----------------------------

"""
Create a Dict mapping (LNFirms, X, S, LFirm) -> row index in A
"""
function index_map(A::Matrix{Int})
    d = Dict{NTuple{4,Int},Int}()
    @inbounds for i in 1:size(A,1)
        d[(A[i,1], A[i,2], A[i,3], A[i,4])] = i
    end
    return d
end

"""
Return indices for all rows in A of the form:
    (n, x, s, 1)
where n = 0:N
"""
function all_incumbent_indices(Amap, N::Int, x::Int, s::Int)
    [Amap[(n, x, s, 1)] for n in 0:N]
end

"""
Return indices for all rows in A of the form:
    (n, x, s, 1) or (n, x, s, 0), depending on LFirm
"""
function pattern_indices(Amap, N::Int, x::Int, s::Int, LFirm::Int)
    [Amap[(n, x, s, LFirm)] for n in 0:N]
end

# ----------------------------
# nfirms 
# ----------------------------
function nfirms(pe1, pi1, bine_e::Vector{Int}, bine_i::Vector{Int}, ne::Int, N::Int)
    # Pe
    Pe = zeros(ne+1)
    @inbounds for j in 1:(ne+1)
        Pe[j] = bine_e[j] * (pe1^(j-1)) * ((1-pe1)^(ne-(j-1)))
    end
    # Pi
    Pi = zeros(N-ne+1)
    @inbounds for k in 1:(N-ne+1)
        Pi[k] = bine_i[k] * (pi1^(k-1)) * ((1-pi1)^(N-ne-(k-1)))
    end

    BigP = zeros(eltype(pe1), N+1)
    @inbounds for j in 0:ne
        for k in 0:(N-ne)
            BigP[j+k+1] += Pe[j+1]*Pi[k+1]
        end
    end
    return BigP
end

# ----------------------------
# prob_entry : equilibrium entry probabilities
# ----------------------------
function prob_entry(Util::Vector{Float64}, trans::Matrix{Float64}, N::Int, Xn::Int, S::Int,
                    bine::Matrix{Int}, Beta::Float64, A::Matrix{Int}, Amap)

    BigP = zeros(Float64, size(A,1), N+1)
    fv   = zeros(Float64, size(A,1))
    eul  = Base.MathConstants.eulergamma

    p  = @. exp(Util)/(1+exp(Util))
    p0 = zeros(size(p))
    p2 = copy(p)

    while maximum(abs.(p0 .- p2)) > 1e-10
        p0 .= p2
        @inbounds for j in 1:size(A,1)
            # incumbents vs entrants row indices
            ind1 = Amap[(min(A[j,1]+A[j,4], N), A[j,2], A[j,3], 0)]
            ind2 = Amap[(max(A[j,1]+A[j,4]-1, 0), A[j,2], A[j,3], 1)]

            BigP[j,:] = nfirms(p[ind1], p[ind2], bine[N-A[j,1]+1, :], bine[A[j,1]+1, :], N-A[j,1], N)

            # fv:
            v = 0.0
            for s2 in 1:S
                # We need log(1 - p(indices for incumbents in next state s2))
                idxs = all_incumbent_indices(Amap, N, A[j,2], s2-1)
                v += trans[A[j,3]+1, s2] * (BigP[j,:]' * log.(1 .- p[idxs]))
            end
            fv[j] = -v

            # tu:
            idxs_util = pattern_indices(Amap, N, A[j,2], A[j,3], A[j,4]) # (0..N, same x,s, LFirm)
            tu = BigP[j,:]' * Util[idxs_util] - Beta*v + Beta*eul
            p[j] = exp(tu)/(1+exp(tu))
        end
        p2 .= p
    end
    fv .= Beta .* (fv .+ eul)
    return p, BigP, fv
end

# ----------------------------
# EntryDataGen : generate entry data via simulation
# ----------------------------
function EntryDataGen(p::Vector{Float64}, ctrans::Matrix{Float64}, S::Int, Xn::Int,
                      Nf::Int, Nm::Int, T::Int, bp::Vector{Float64},
                      Tl::Int, A::Matrix{Int}, Amap)

    Firm  = zeros(Bool, Nm, T+Tl, Nf)
    Lfirm = zeros(Bool, Nm, T+Tl+1, Nf)
    X     = rand(0:Xn-1, Nm)          # 0..Xn-1
    State = zeros(Int, Nm, T+Tl+1)
    Y     = zeros(Float64, Nm, T+Tl)

    State[:,1] = rand(1:S, Nm)        
    Draw1 = rand(Nm, T+Tl, Nf)
    Draw2 = rand(Nm, T+Tl)
    Draw3 = randn(Nm, T+Tl)

    for nm in 1:Nm
        Nfirm = 0
        for t in 1:(T+Tl)
            for nf in 1:Nf
                ind = Amap[(Nfirm - Int(Lfirm[nm,t,nf]), X[nm], State[nm,t]-1, Int(Lfirm[nm,t,nf]))]
                Firm[nm,t,nf] = p[ind] > Draw1[nm,t,nf]
            end
            Nfirm = count(Firm[nm,t,:])
            Lfirm[nm,t+1,:] = Firm[nm,t,:]

            # Y
            Y[nm,t] = [1.0, Nfirm, X[nm]-1, State[nm,t]-1]' ⋅ bp + Draw3[nm,t]

            # next state:
            State[nm,t+1] = 1
            for s in 1:(S-1)
                State[nm,t+1] += (Draw2[nm,t] > ctrans[State[nm,t], s]) ? 1 : 0
            end
        end
    end

    # Trim burn-in:
    Firm  = Firm[:, Tl+1:T+Tl, :]
    State = State[:, Tl+1:T+Tl] .- 1
    Lfirm = Lfirm[:, Tl+1:T+Tl, :]
    Y     = Y[:, Tl+1:T+Tl]

    return Firm, X, State, Y, Lfirm
end

# ===================================================
# Main script body
# ===================================================

function main()
    # Parameters
    Nm   = 3000
    Nf   = 5
    T    = 10
    Tl   = 10
    S    = 5
    Xn   = 10
    Beta = 0.9

    # structural entry coefficients theta:
    bx  = -0.05
    bx0 = 0.0
    bnf = -0.2
    bss = 0.25
    be  = -1.5

    # Price coefficients -- not used until unobserved states:
    bp = [7.0; -0.4; -0.1; 0.3]

    # Transition of aggregate state
    ps  = 0.7
    nps = (1-ps)/(S-1)
    trans = ps*I(S) .+ nps*(ones(S,S) .- I(S)) |> Matrix{Float64}

    # Construct ctrans exactly as MATLAB cumulative:
    ctrans = trans[:,1]
    for s in 1:S-2
        ctrans = hcat(ctrans, ctrans[:,s] .+ trans[:,s+1])
    end
    # ctrans is S x (S-1) cumulative; matches your use in EntryDataGen.

    nf = collect(0:Nf)

    # Build state space A:
    # state_args = {0:Nf, 0:Xn-1, 0:S-1, 0:1}
    A0 = collect(0:Nf)
    A1 = collect(0:Xn-1)
    A2 = collect(0:S-1)
    A3 = collect(0:1)
    A = Array{Int}(undef, (Nf+1)*Xn*S*2, 4)
    # Equivalent to ndgrid then reshape in MATLAB:
    idx = 1
    for a0 in A0, a1 in A1, a2 in A2, a3 in A3
        A[idx,1] = a0
        A[idx,2] = a1
        A[idx,3] = a2
        A[idx,4] = a3
        idx += 1
    end

    # bine matrix:
    bine = zeros(Int, Nf+1, Nf+1)
    for n in 1:(Nf+1)
        for k in 1:n
            bine[n,k] = binomial(n-1, k-1)
        end
    end

    theta = [bnf, bx, bss, be]
    Util  = zeros(Float64, size(A,1))
    @inbounds for j in 1:size(A,1)
        Util[j] = bx*A[j,2] + bnf*A[j,1] + bss*A[j,3] + be*(1 - A[j,4])
    end

    N = Nf

    # --- Fixed point for equilibrium CCPs ---
    Amap = index_map(A)
    p2, BigP_fp, fv_fp = prob_entry(Util, trans, Nf, Xn, S, bine, Beta, A, Amap)

    Random.seed!(19)

    # --- Simulate data ---
    Firm, X, State, Y, Lfirm = EntryDataGen(p2, ctrans, S, Xn, Nf+1, Nm, T, bp, Tl, A, Amap)

    # ---------------------------------
    # Reshape data 
    # ---------------------------------
    # S2, X2 replication
    S2 = repeat(State, inner=(1,1,Nf+1))
    S2 = vec(S2)
    X2 = repeat(X, inner=(1,T,Nf+1))
    X2 = vec(X2)

    # NFirm and LNFirm
    NFirm = sum(Firm, dims=3)
    LNFirm = sum(Lfirm, dims=3)
    NFirm2 = vec(repeat(NFirm, inner=(1,1,Nf+1)))
    LNFirm2 = vec(repeat(LNFirm, inner=(1,1,Nf+1)))

    Firm2 = vec(Firm)
    LFirm2 = vec(Lfirm)

    LNFirm2 .= LNFirm2 .- LFirm2

    # Z
    Z = hcat(ones(Float64,(Nf+1)*Nm*T), X2, S2, 1 .- LFirm2, LNFirm2)

    # -------------------------------
    # Reduced-form CCP estimation
    # -------------------------------
    W_ccp = hcat(
        X2,
        (X2 ./ 10).^2,
        LFirm2,
        LNFirm2,
        (LNFirm2 ./ 5).^2,
        S2,
        S2 .* X2 ./ 10,
        LFirm2 .* S2,
        LNFirm2 .* S2 ./ 10
    )

    Firm3 = Firm2 .> 0.5
    # Put into DataFrame for GLM:
    df_ccp = DataFrame(Firm3 = Firm3)
    for j in 1:size(W_ccp,2)
        df_ccp[!, Symbol("W$j")] = W_ccp[:,j]
    end

    # Fit: logit (binary), just like mnrfit with 2 cats
    form_terms = join(["W$j" for j in 1:size(W_ccp,2)], " + ")
    f = eval(Meta.parse("@formula(Firm3 ~ $(form_terms))"))
    model = glm(f, df_ccp, Binomial(), LogitLink())
    λ = coef(model)
    # Build design for A grid ("B_fit" in MATLAB)
    B = DataFrame(S = A[:,3], X = A[:,2], LNFirms = A[:,1], LFirm = A[:,4])
    B_fit = hcat(
        B.X,
        (B.X ./ 10).^2,
        B.LFirm,
        B.LNFirms,
        (B.LNFirms ./ 5).^2,
        B.S,
        B.S .* B.X ./ 10,
        B.LFirm .* B.S,
        B.S .* B.LNFirms ./ 10
    )

    # ccp_hat: logistic link inverse
    # Add intercept column of 1’s:
    X_design = hcat(ones(size(B_fit,1)), B_fit)
    ccp_hat = 1 ./(1 .+ exp.(-(X_design * λ)))  # probability of entry=1 (Firm3==true)
    p = ccp_hat
    # bigp, fv (second pass using estimated CCPs)
    bigp = zeros(Float64, size(A,1), N+1)
    fv   = zeros(Float64, size(A,1))
    eul  = Base.MathConstants.eulergamma
    for j in 1:size(A,1)
        ind1 = Amap[(min(B.LNFirms[j] + B.LFirm[j], N), B.X[j], B.S[j], 0)]
        ind2 = Amap[(max(B.LNFirms[j] + B.LFirm[j] - 1, 0), B.X[j], B.S[j], 1)]
        pe1  = p[ind1]
        pi1  = p[ind2]
        ne   = N - B.LNFirms[j]

        bine_temp = bine[ne+1, :]
        bini_temp = bine[B.LNFirms[j]+1, :]

        # Build Pe, Pi
        Pe = bine_temp[1:ne+1] .* (pe1 .^ ((1:ne+1).-1)) .* ((1-pe1) .^ (ne .- ((1:ne+1).-1)))
        Pi = bini_temp[1:N-ne+1] .* (pi1 .^ ((1:N-ne+1).-1)) .* ((1-pi1) .^ (N-ne .- ((1:N-ne+1).-1)))

        BigP_temp = zeros(Float64, N+1)
        @inbounds for i in 0:ne
            for k in 0:(N-ne)
                BigP_temp[i+k+1] += Pe[i+1] * Pi[k+1]
            end
        end
        bigp[j,:] .= BigP_temp

        # FV
        v = 0.0
        for s2 in 1:S
            idxs = all_incumbent_indices(Amap, N, B.X[j], s2-1)
            v   += trans[B.S[j]+1, s2] * (bigp[j,:]' * log.(1 .- p[idxs]))
        end
        fv[j] = -v
    end
    fv .= Beta .* (fv .+ eul)

    # Match state realizations:
    # Instead of 'ismember' in Matlab, we'll loop & use Amap
    obs = size(Z,1)
    state_ind = Vector{Int}(undef, obs)
    @inbounds for i in 1:obs
        state_ind[i] = Amap[(LNFirm2[i], Z[i,2], Z[i,3], LFirm2[i])]
    end
    FV   = fv[state_ind]
    BigP = bigp[state_ind, :]

    # last column of Z becomes BigP * nf
    Z[:,end] .= BigP * nf

    # Likelihood
    function LikeFun(b::AbstractVecOrMat)
        η = Z * b .+ FV
        # sum(log(1+exp(η)) - Firm2.*η)
        return sum(log1pexp.(η)) - dot(Firm2, η)
    end

    # Optimize 
    b0 = 0.1*ones(5)
    res = optimize(LikeFun, b0, LBFGS(); autodiff = :forward)
    ThetaHat = Optim.minimizer(res)

    # EntryEstimatesTable
    rows = ["Constant","Num. of competitors","Constant state x","Time-varying state s","Entry cost"]
    CCP   = ThetaHat[[1,5,2,3,4]]
    TrueV = [0.0; theta]
    EntryEstimatesTable = DataFrame(
        Row = rows,
        CCP_estimates = CCP,
        True_values   = TrueV
    )

    println(EntryEstimatesTable)

    # Visualize convergence (optional)
    Δvec = 0.9:0.025:1.1
    llvals = [LikeFun((1+Δ).*ThetaHat) for Δ in Δvec]
    scatter(1 .+ collect(Δvec), llvals,
            xlabel = "1+Δ",
            ylabel = "Negative log-likelihood",
            title  = "Log-likelihood as a function of (1+Δ)θ")

    return (;ThetaHat, EntryEstimatesTable, p2, BigP_fp, fv_fp, p, bigp, fv)
end

# Run if script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
