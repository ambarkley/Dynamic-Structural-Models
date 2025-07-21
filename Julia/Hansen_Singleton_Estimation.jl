##############################################
# Two–asset GMM estimation of Hansen &       #
# Singleton (1982).                          #
#                                            #
# Author: Aaron Barkley                      #
# Date  : July 2025                          #
##############################################

using CSV, DataFrames, Dates, HTTP
using Statistics, LinearAlgebra, Distributions
using Optim, ForwardDiff

"""
FUNCTIONS

These (i) read data and aggregate to quarters and
(ii) define Newey-West weight function
"""


"Download a FRED series as a DataFrame (DATE,value)."
function fred_csv(series::String)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$(series)"
    io  = HTTP.get(url).body |> IOBuffer
    df  = CSV.read(io, DataFrame; dateformat = dateformat"yyyy-mm-dd")
    rename!(df, [:observation_date, :value])
    df.observation_date = Date.(df.observation_date)               # convert to Date type
    return df
end

"Return quarter‑end Date for any daily/monthly Date."
quarter_end(d::Date) = Date(year(d), month(d) <= 3 ? 3 : month(d) <= 6 ? 6 : month(d) <= 9 ? 9 : 12, 1) |> x -> Dates.lastdayofmonth(x)

"Aggregate to quarterly using function f (mean,last)."
function to_quarterly(df::DataFrame; f = mean)
    df.qrt_end = quarter_end.(df.observation_date)
    g = groupby(df, :qrt_end)
    out = combine(g, :observation_date => first => :observation_date, :value => f => :value)
    sort!(out, :observation_date)
    return out
end

"Simple Newey–West HAC covariance (lag L).  umat is T×k matrix of moment obs."
function newey_west(umat::Matrix{Float64}, L::Int)
    T, k = size(umat)
    Σ = umat' * umat / T                     # lag 0
    for ℓ in 1:L
        w = 1.0 - ℓ/(L + 1)
        Γ = umat[(ℓ+1):end, :]' * umat[1:(end-ℓ), :] / T
        Σ += w * (Γ + Γ')
    end
    return Σ
end

"""
Read raw series  
"""

# If readLocal == true, read data from local folder (uses value-weighted returns including dividends)
# Otherwise, reads all data from FRED and uses risk-free rate and NASDAQ price-only returns as assets
readLocal = false

if readLocal == false 
    pce      = fred_csv("PCECC96")        # quarterly already
    bill_raw = fred_csv("DTB3")           # daily, % discount
    nasdaq_raw   = fred_csv("NASDAQCOM")          # daily, index level
    cpi_raw  = fred_csv("CPIAUCSL")       # monthly CPI (SA)
        
    for j=2:size(nasdaq_raw,1)
        if ismissing(nasdaq_raw.value[j])
            nasdaq_raw.value[j] = nasdaq_raw.value[j-1]
        end
    end

else
    pce  = CSV.read("\\..\\Data\\PCECC96.csv", DataFrame; dateformat = dateformat"yyyy-mm-dd")
    bill_raw  = CSV.read("..\\..\\Data\\DTB3.csv", DataFrame; dateformat = dateformat"yyyy-mm-dd")
    sp_raw =  CSV.read("..\\..\\Data\\SP500.csv", DataFrame; dateformat = dateformat"yyyy-mm-dd")
    cpi_raw = CSV.read("..\\..\\Data\\CPIAUCSL.csv", DataFrame; dateformat = dateformat"yyyy-mm-dd")
    equity_raw = CSV.read("..\\..\\Data\\crsp_agg_monthly_total_returns.csv", DataFrame; dateformat = dateformat"yyyy-mm-dd")

    vwr_raw = equity_raw[:,[1,2]]
    ewr_raw = equity_raw[:, [1,3]]
    rename!(vwr_raw, :DATE => :observation_date, :vwretd => :value)
    rename!(ewr_raw, :DATE => :observation_date, :ewretd => :value)

    rename!(pce, :PCECC96 => :value)
    rename!(bill_raw, :DTB3 => :value)
    rename!(sp_raw, :SP500 => :value)
    rename!(cpi_raw, :CPIAUCSL => :value)
end



# fix missing in data

for j=2:size(bill_raw,1)
    if ismissing(bill_raw.value[j])
        bill_raw.value[j] = bill_raw.value[j-1]
    end
end



# Quarterly aggregation

bill_q  = to_quarterly(bill_raw; f = mean)
sp_q    = to_quarterly(sp_raw;   f = last)
cpi_q   = to_quarterly(cpi_raw;  f = last)
if readLocal==false
    nasdaq_q = to_quarterly(nasdaq_raw; f=mean)
    nasdaq_q.observation_date = Date.(year.(nasdaq_q.observation_date), month.(nasdaq_q.observation_date), 1)

else
    vwr_q = to_quarterly(vwr_raw; f=last)
    ewr_q = to_quarterly(ewr_raw; f=last)
    vwr_q.observation_date = Date.(year.(vwr_q.observation_date), month.(vwr_q.observation_date), 1)
    ewr_q.observation_date = Date.(year.(ewr_q.observation_date), month.(ewr_q.observation_date), 1)
end

"""
Merge and apply time window to data  
"""

bill_q.observation_date = Date.(year.(bill_q.observation_date), month.(bill_q.observation_date), 1)
cpi_q.observation_date = Date.(year.(cpi_q.observation_date), month.(cpi_q.observation_date), 1)


df = innerjoin(pce, bill_q, on = :observation_date, makeunique = true)
df = innerjoin(df, cpi_q, on = :observation_date, makeunique = true)
if readLocal == false
    df = innerjoin(df, nasdaq_q, on = :observation_date, makeunique = true)
    df = df[:,[1,2,4,6,8]];
    rename!(df, [:observation_date, :pce, :disc, :cpi, :nsdq])
    startDate = Date(1972,1,1)

else
    df = innerjoin(df, vwr_q, on = :observation_date, makeunique = true)
    df = innerjoin(df, ewr_q, on = :observation_date, makeunique = true)
    df = df[:,[1,2,4,6,8, 10]];
    rename!(df, [:observation_date, :pce, :disc, :cpi, :vwr, :ewr])
    df.vwr = 1.0 .+ df.vwr
    df.ewr = 1.0 .+ df.ewr

    startDate = Date(1960,1,1)
end

# user‑chosen sample window
startDate = Date(1960,1,1)
endDate   = Date(2019,12,31)
filter!(row -> startDate ≤ row.observation_date ≤ endDate, df)

"""
Create variables
"""

logC = log.(df.pce)
ΔlogC = diff(logC)   # g_{t+1}, length N-1

# risk‑free return (nominal) observed at t (disc in decimals)
disc = df.disc[1:end-1] ./ 100.0
Rfnom = 1.0 ./ (1 .- disc .* 90.0/360.0)



if readLocal==false
    nsdq  = df.nsdq
    Renom = Rfnom # set second equity to the risk-free asset if reading web data only
    Revnom = nsdq[2:end] ./ nsdq[1:end-1] # first asset is NASDAQ price-only returns
else
    Renom = df.ewr[2:end]
    Revnom = df.vwr[2:end]
end
# inflation factor π_t = CPI_{t+1}/CPI_t
pi  = df.cpi[2:end] ./ df.cpi[1:end-1]

# real returns
Rf_all = Rfnom ./ pi
Re_all = Renom ./ pi
Rv_all = Revnom ./ pi

ΔlogC = 1.0 .+ ΔlogC 

# align lengths (drop first observation)
gC_lead = ΔlogC                                
n = length(gC_lead) - 1

gLag = gC_lead[1:n]                            
gC   = gC_lead[2:end]                          
Rf_lag   = Rf_all[1:n] 
Re_lag   = Re_all[1:n]
Rv_lag = Rv_all[1:n]

Rf = Rf_all[2:end]
Re = Re_all[2:end]
Rv = Rv_all[2:end]

# instrument matrix Z_t: constant + g_t
X = hcat(ones(n), Re_lag, Rv_lag, gLag)                        
L = size(X,2)
    
"""
Define Moment Fuction and GMM Criterion
"""

"mfun(theta) -> n×2 matrix of pricing errors (risk‑free, equity)."
function mfun(theta::AbstractVector)
    β, γ = theta
    sdf = β.*(gC.^(-γ))
    m1 = sdf .* Re .- 1.0
    m2 = sdf .* Rv .- 1.0
    return hcat(m1, m2)                       # n × 2
end

function stacked_u(theta)
    m = mfun(theta)                            # n × 2
    u1 = X .* m[:,1]                           # n × L
    u2 = X .* m[:,2]
    return hcat(u1, u2)                       # n × (2L)
end

"Sample average of stacked moments, 1×(2L) row vector."
gbar(theta) = mean(stacked_u(theta), dims = 1) |> vec |> transpose

"GMM objective given weight matrix W"
function Q(theta, W)
    g = gbar(theta)
    return (g * W * g')[1]
end


"""
Two‑step GMM         
"""
θ_0 = [0.99, 5]                              

# ---- First step: identity weight ----
W1 = I(2L)
obj1 = θ -> Q(θ, I)
res1 = optimize(obj1, θ_0, BFGS(); autodiff = :forward)
#res1 = optimize(obj1, θ_0, BFGS())
θ1  = Optim.minimizer(res1)

# ----  Newey–West optimal weight ----
U  = stacked_u(θ1)
S  = newey_west(Matrix{Float64}(U), 4)        
W2 = inv(S)
obj2 = θ -> Q(θ, W2)
res2 = optimize(obj2, θ1, BFGS())
θ_hat = Optim.minimizer(res2)

# ---- asymptotic covariance & J‑test ----
D = ForwardDiff.jacobian(t -> gbar(t)', θ_hat)    
Vθ = inv(D' * W2 * D) / n                     
se = sqrt.(diag(Vθ))

g = gbar(θ_hat)
J = n * (g * W2 * g')[1]
println("\n===== Hansen & Singleton two‑asset GMM (quarterly) =====")
println(rpad("beta",8),  round(θ_hat[1], digits=4), "    se = ", round(se[1], digits=4))
println(rpad("gamma",8), round(θ_hat[2], digits=4), "    se = ", round(se[2], digits=4),
        "   (alpha = ", round(-θ_hat[2], digits=4), ")")
println("J‑stat = ", round(J, digits=2), "  (df = ", 2L - 2, ")  p = ", round(1 - cdf(Chisq(2L-2), J), digits=3))
