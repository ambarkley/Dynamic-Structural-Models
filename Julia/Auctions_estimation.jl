using Random, Distributions, Optim, Plots, QuadGK, StatsBase, KernelDensity, NLsolve, Polynomials

"""
First-price sealed bid auction
"""
mu_b = 7.5  # Average bidder's valuation
sigma_b = 0.75  # Standard deviation of bidder valuation distribution.
BuyerValueDist = Normal(mu_b, sigma_b)

num_auc = 1000  # Simulate 1000 auctions
N = 5  # Assume five bidders at each auction
Random.seed!(2023)

# Equilibrium bid function:
B(v) = v .- ((1.0 ./ cdf(BuyerValueDist, v)).^(N - 1)) .* quadgk(u -> cdf(BuyerValueDist, u).^(N - 1), 0.0, v)[1]

# Draw valuations from the distribution defined above:
value_draws = rand(BuyerValueDist, num_auc, N)
bids = B.(value_draws[:])

scatter(value_draws[:], bids, xlabel = "Valuation", ylabel = "Bids", label = "Simulated bids")
plot!(value_draws[:], value_draws[:], linestyle = :dash, color = :black, label = "45 degree line")


G= ecdf(bids)  # Empirical cdf
X=range(4.5,8,100)
g = kde(bids)  # Kernel density estimation of pdf
pseudo_values = bids .+ G.(bids) ./ ((N - 1) .* pdf(g,bids))

f = kde(pseudo_values)

v = 5.0:0.1:10.0
plot(v, pdf(f,v), lw=3, label = "Estimated value distribution")
plot!(v, pdf.(BuyerValueDist, v), lw=3, ls = :dash, label = "True value distribution")

log_likelihood(theta) =  -sum(logpdf.(Normal(theta[1], theta[2]), pseudo_values))
   

est_result = optimize(log_likelihood, [9.0, 1.0])
theta_hat=est_result.minimizer
w = range(5, stop = 10, length = 100)
plot(w, pdf.(Normal(theta_hat[1], theta_hat[2]), w), linestyle = :dash, color = :black, label = "Estimated distribution - Normal parameterization")
plot!(v, pdf.(BuyerValueDist, v), linewidth = 2, label = "True distribution")
plot!(legend = :best)


"""
Second-price sealed bid auction
"""

mu_b = 7.5  # Average bidder's home valuation, in $100,000's.
sigma_b = 0.75  # Standard deviation of bidder valuation distribution.
BuyerValueDist = Normal(mu_b, sigma_b)  # Assume a normal distribution for valuations

num_auc = 1000  # Simulate 2000 auctions
N = 3  # Assume two bidders at each auction
Random.seed!(19)

# Draw valuations from the distribution defined above:
value_draws = rand(BuyerValueDist, num_auc, N)

WBids_second = sort(value_draws, dims = 2, rev = true)[:, 2]
F_V2= ecdf(WBids_second)
F_V = zeros(length(v))

F_V = nlsolve(u -> N .* (u.^(N - 1)) .* (1 .- u) .+ u.^N .- F_V2.(v), F_V2.(v)).zero

plot(v, F_V, linestyle = :solid, color = :blue, label = "Nonparametric estimates")
plot!(v, cdf.(BuyerValueDist, v), linestyle = :dash, color = :black, label = "True distribution")
xlims!((5.5, 9))
xlabel!("Valuation")
ylabel!("F_V")
