#------------------------------------------------------------
# Computational Econ 899b
# Problem Set 5: Besanko and Doraszelski (2004)
# Authors: Jackson Crawford, Dalya Elmalt, Jon Kroah
# Referenced code from Alex von Hafften and John Higgins.
#------------------------------------------------------------

using Parameters, DataFrames, LatexPrint, Plots, Random, Distributions

# Struct to hold model primitives
@with_kw struct Primitives
    δ::Float64
    α::Float64
    β::Float64
    a::Float64
    b::Float64

    q_grid_min::Float64
    q_grid_max::Float64
    q_grid_step::Float64
    q_grid::Array{Float64}
    q_grid_length::Int64

    ω::Array{Tuple{Float64, Float64}} = [(0.0, 0.0)]
    ω_length::Int64

    ω_paths_T::Int64
    ω_paths_reps::Int64
end

# Struct to hold results_LBFGS
mutable struct Results
    q_star_1::Array{Float64, 2}
    q_star_2::Array{Float64, 2}
    π_star_1::Array{Float64, 2}
    π_star_2::Array{Float64, 2}

    valfunc_1::Array{Float64, 2}
    valfunc_2::Array{Float64, 2}
    
    x_polfunc_1::Array{Float64, 2}
    x_polfunc_2::Array{Float64, 2}

    Q_prob::Array{Float64, 2}
    
    ω_paths_sim::Array{Int64, 2}
end

# Create a struct with values for all the primitives
function initialize()
    # Model primitives
    δ = 0.1
    β = 1 / 1.05
    α = 0.06
    a = 40
    b = 10

    q_grid_min = 0.0
    q_grid_max = 45.0
    q_grid_step = 5.0
    q_grid = q_grid_min:q_grid_step:q_grid_max
    q_grid_length = length(q_grid)

    ω_paths_T = 25
    ω_paths_reps = 1000
    
    ω = [(0.0, 0.0)]
    count = 1
    for i_1=1:q_grid_length, i_2=1:q_grid_length
        if count == 1
            ω[count] = (q_grid[i_1], q_grid[i_2])
        else
            push!(ω, (q_grid[i_1], q_grid[i_2]))
        end
        count += 1
    end
    ω_length = length(ω)
    @assert ω_length == q_grid_length^2 "Error: ω is wrong length; should be q_grid_length²."

    prim = Primitives(δ, α, β, a, b, q_grid_min, q_grid_max, q_grid_step, q_grid, q_grid_length, ω, ω_length, ω_paths_T, ω_paths_reps)

    # Results
    q_star_1 = zeros(q_grid_length, q_grid_length)
    q_star_2 = zeros(q_grid_length, q_grid_length)
    
    π_star_1 = zeros(q_grid_length, q_grid_length)
    π_star_2 = zeros(q_grid_length, q_grid_length)

    valfunc_1 = zeros(q_grid_length, q_grid_length)
    valfunc_2 = zeros(q_grid_length, q_grid_length)

    x_polfunc_1 = zeros(q_grid_length, q_grid_length)
    x_polfunc_2 = zeros(q_grid_length, q_grid_length)

    Q_prob = zeros(q_grid_length^2, q_grid_length^2)

    ω_paths_sim = Int64.(zeros(ω_paths_T, ω_paths_reps))

    results = Results(q_star_1, q_star_2, π_star_1, π_star_2, valfunc_1, valfunc_2, x_polfunc_1, x_polfunc_2, Q_prob, ω_paths_sim)

    return prim, results
end

# Compute inverse demand (market-clearing price)
function inv_demand(Q::Float64, prim::Primitives)
    @unpack a, b = prim

    p = (a / b) - (1 / b) * Q
    return p
end

# Compute static payoffs
function static_payoffs(q_1, q_2, prim)
    p = inv_demand(q_1 + q_2, prim)

    π_1 = p * q_1
    π_2 = p * q_2

    return π_1, π_2
end

# Compute static Cournot Nash equilibria (q_star_1, q_star_2)
# INPUTS: industry state (q_bar_1, q_bar_2), model primitives; tolerance level; update factor
# OUTPUTS: Nash equilibrium quantities (q_star_1, q_star_2)
function cournot_eq(q_bar_1::Float64, q_bar_2::Float64, prim::Primitives, tol::Float64, update_factor::Float64)

    @unpack a, b = prim

    # println("Solving Cournot Nash equilibrium. Capacities are ($q_bar_1, $q_bar_2).")

    # First guess: optimal to play your capacity
    q_star_1 = q_bar_1
    q_star_2 = q_bar_2

    converge_flag = 0
    count = 0
    while converge_flag == 0
        # println("Solving Cournot: iteration $count. Current guess for NE: ($q_star_1, $q_star_2)")

        q_star_1_old = q_star_1
        q_star_2_old = q_star_2

        # this part is right if demand in the pset is not inverse demand
        q_star_1_update = max(0.0, min(q_bar_1, (a - q_star_2) / 2))
        q_star_2_update = max(0.0, min(q_bar_2, (a - q_star_1) / 2))

        # this part is right if demand in the pset is INVERSE demand
        # q_star_1_update = max(0.0, min(q_bar_1, (a - b * q_star_2) / (2 * b)))
        # q_star_2_update = max(0.0, min(q_bar_2, (a - b * q_star_1) / (2 * b)))

        err = max(abs(q_star_1 - q_star_1_update), abs(q_star_2 - q_star_2_update))
        if err >= tol
            q_star_1 = (1 - update_factor) * q_star_1_old + update_factor * q_star_1_update
            q_star_2 = (1 - update_factor) * q_star_2_old + update_factor * q_star_2_update
            count += 1
        else
            converge_flag = 1
            # println("Converged to Cournot NE in $count iterations. NE strategies = ($q_star_1, $q_star_2)")
        end
    end

    return q_star_1, q_star_2
end

# Solve Cournot equilibria for all possible industry states
# INPUT: industry states (q_grid), results
# OUTPUT: none; modifies Results struct
function solve_cournot(prim::Primitives, results::Results; tol::Float64=1e-12, update_factor::Float64=0.5)
    @unpack q_grid = prim
    for (i_1, q_bar_1) in enumerate(q_grid)
        for (i_2, q_bar_2) in enumerate(q_grid)
            q_star_1, q_star_2 = cournot_eq(q_bar_1, q_bar_2, prim, tol, update_factor)
            π_star_1, π_star_2 = static_payoffs(q_star_1, q_star_2, prim)

            results.q_star_1[i_1, i_2] = q_star_1
            results.q_star_2[i_1, i_2] = q_star_2

            results.π_star_1[i_1, i_2] = π_star_1
            results.π_star_2[i_1, i_2] = π_star_2
        end
    end
end

# Markov switching probability for individual firm's capacity
# INPUTS: "from" state (q_bar), "to" state (q_bar_p), investment (x), primitives (prim)
# OUTPUT: transition probability
function pr_capacity_transition(q_bar::Float64, q_bar_p::Float64, x::Float64, prim::Primitives)
    @unpack δ, α, q_grid_min, q_grid_max, q_grid_step = prim

    # Transition probs at boundary of q_bar support:
    # (See Besanko and Doraszelski for formulas)
    if q_bar == q_grid_min
        if q_bar_p == q_bar
            prob = 1 / (1 + α * x)
        elseif q_bar_p == q_bar + q_grid_step
            prob = α * x / (1 + α * x)
        else
            prob = 0.0
        end
    elseif q_bar == q_grid_max
        if q_bar_p == q_bar
            prob = (1 - δ + α * x) / (1 + α * x)
        elseif q_bar_p == q_bar - q_grid_step
            prob = δ / (1 + α * x)
        else
            prob = 0.0
        end
    # Transition probs in interior of q_bar support:
    else
        if q_bar_p == q_bar
            prob = ((1 - δ) + δ * α * x) / (1 + α * x)
        elseif q_bar_p == q_bar + q_grid_step
            prob = ((1 - δ) * α * x) / (1 + α * x)
        elseif q_bar_p == q_bar - q_grid_step
            prob = δ / (1 + α * x)
        else
            prob = 0.0
        end
    end

    if (q_bar_p < q_grid_min) || (q_bar_p > q_grid_max)
        prob = 0.0
    end

    return prob
end

# Firm 1's "W" function
# INPUTS:
#   valfunc (Results), q_grid (Primitives), firm 1 state (q_bar_1),
#   firm 2 "from" state (q_bar_2), firm 2 investment (x_2)
# OUTPUT:
#   W function
#   = expected continuation value for firm 1, taken over firm 2's capacity next period
function W(q_bar_1::Float64, q_bar_2::Float64, x_2::Float64, prim::Primitives, valfunc_1::Array{Float64})
    @unpack q_grid = prim

    i_q_1 = findfirst(x->x==q_bar_1, q_grid)

    W = 0.0
    for (i_q_2_p, q_bar_2_p) in enumerate(q_grid)
        W += valfunc_1[i_q_1, i_q_2_p] * pr_capacity_transition(q_bar_2, q_bar_2_p, x_2, prim)
    end

    return W
end


# Bellman operator (do firms 1 & 2 at the same time)
# (Referred to Alex von Hafften's code here)
# INPUTS:
#   Current guess for valfunc (Results)
#   Model primitives (Primitives)
# OUTPUT:
#   Updated guess for valfunc (nq * nq matrix)
function bellman(prim::Primitives, results::Results; tol::Float64=1e-6)
    @unpack valfunc_1, valfunc_2, q_star_1, q_star_2, π_star_1, π_star_2 = results
    @unpack q_grid, q_grid_step, q_grid_length, α, β, δ = prim

    # Solve problem for every state (q_bar_1, q_bar_2)
    valfunc_1_update = zeros(q_grid_length, q_grid_length)
    valfunc_2_update = zeros(q_grid_length, q_grid_length)

    x_polfunc_1_update = zeros(q_grid_length, q_grid_length)
    x_polfunc_2_update = zeros(q_grid_length, q_grid_length)

    for (i_q_1, q_bar_1) in enumerate(q_grid), (i_q_2, q_bar_2) in enumerate(q_grid)
        
        # Fixing x_2, find optimal x_1 (using formula in Besanko & Doraszelski)
        # Note, we do not need to iterate this optimization problem (as in the Cournot solver above)
        #   *inside* the Bellman operator. The optimizers (x_1_star, x_2_star) will converge as we
        #   apply the Bellman operator iteratively and will yield the same result as if we iterated
        #   inside the Bellman operator (I checked).
        vf_1 = results.valfunc_1

        W_1_nochange    = W(q_bar_1, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, vf_1)
        
        if i_q_1 == 1            # capacity can't decrease if currently at the min
            W_1_down        = W_1_nochange
        else
            W_1_down        = min(W(q_bar_1 - q_grid_step, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, vf_1), W_1_nochange)
        end
        
        if i_q_1 == q_grid_length # capacity can't increase if currently at the max
            W_1_up          = W_1_nochange
        else
            W_1_up          = max(W(q_bar_1 + q_grid_step, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, vf_1), W_1_nochange)
        end

        x_polfunc_1_update[i_q_1, i_q_2] = max(
            0.0,
            (-1.0 + sqrt(β * α * ((1 - δ) * (W_1_up - W_1_nochange) + δ * (W_1_nochange - W_1_down)))) / α
        )

        valfunc_1_update[i_q_1, i_q_2] = π_star_1[i_q_1, i_q_2] - x_polfunc_1_update[i_q_1, i_q_2]
        for (i, q_bar_1_p) in enumerate(q_grid)
            valfunc_1_update[i_q_1, i_q_2] += β * W(q_bar_1_p, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, vf_1) * pr_capacity_transition(q_bar_1, q_bar_1_p, x_polfunc_1_update[i_q_1, i_q_2], prim)
        end
        

        # Fixing x_1, find optimal x_2 (using formula in Besanko & Doraszelski)
        #   NOTE: Need to transpose firm 2's value function (matrix), b/c o/w rows are opponent's state (q_bar_1)
        #       and columns are own state (q_bar_2). This is b/c the W function (defined above) needs the rows of
        #       the value function to be *own* state & cols to be *opponent's* state.
        vf_2 = Float64.(results.valfunc_2')
        
        W_2_nochange    = W(q_bar_2, q_bar_1, results.x_polfunc_1[i_q_1, i_q_2], prim, vf_2)
        
        if i_q_2 == 1 # capacity can't decrease if at the min
            W_2_down        = W_2_nochange
        else
            W_2_down        = min(W(q_bar_2 - q_grid_step, q_bar_1, results.x_polfunc_1[i_q_1, i_q_2], prim, vf_2), W_2_nochange)
        end
        
        if i_q_2 == q_grid_length # capacity can't increase if at the max
            W_2_up          = W_2_nochange
        else
            W_2_up          = max(W(q_bar_2 + q_grid_step, q_bar_1, results.x_polfunc_1[i_q_1, i_q_2], prim, vf_2), W_2_nochange)
        end

        x_polfunc_2_update[i_q_1, i_q_2] = max(
            0.0,
            (-1.0 + sqrt(β * α * ((1 - δ) * (W_2_up - W_2_nochange) + δ * (W_2_nochange - W_2_down)))) / α
        )

        valfunc_2_update[i_q_1, i_q_2] = π_star_2[i_q_1, i_q_2] - x_polfunc_2_update[i_q_1, i_q_2]
        for (i, q_bar_2_p) in enumerate(q_grid)
            valfunc_2_update[i_q_1, i_q_2] += β * W(q_bar_2_p, q_bar_1, results.x_polfunc_1[i_q_1, i_q_2], prim, vf_2) * pr_capacity_transition(q_bar_2, q_bar_2_p, x_polfunc_2_update[i_q_1, i_q_2], prim)
        end

        # println("(q_bar_1, q_bar_2) = ($q_bar_1, $q_bar_2): x_1_star = $(x_polfunc_1_update[i_q_1, i_q_2]), x_2_star = $(x_polfunc_2_update[i_q_1, i_q_2])")
    end

    return valfunc_1_update, valfunc_2_update, x_polfunc_1_update, x_polfunc_2_update
end

# Value function iteration
function iterate_bellman(prim::Primitives, results::Results; tol::Float64=1e-12, maxiter::Int64=10_000)
    conv_flag = 0
    iter = 1
    while conv_flag == 0
        # Apply Bellman operator once
        valfunc_1_update, valfunc_2_update, x_polfunc_1_update, x_polfunc_2_update = bellman(prim, results)

        # Max distance between new and old value functions
        err = max(maximum(abs.(results.valfunc_1 .- valfunc_1_update)), maximum(abs.(results.valfunc_2 .- valfunc_2_update)))
        
        if iter == maxiter && err >= tol
            println("Value function iteration failed to converged in $iter iterations. Error = $err, tol = $tol.")
            conv_flag = 1
        elseif iter <= maxiter && err < tol
            conv_flag = 1
            println("Value function iteration converged in $iter iterations! Error = $err, tol = $tol.")
        elseif iter <= maxiter && err >= tol
            # println("---Value function iteration $iter: error = $err")
        end

        # Update policy functions
        results.x_polfunc_1 = x_polfunc_1_update
        results.x_polfunc_2 = x_polfunc_2_update

        results.valfunc_1 = valfunc_1_update
        results.valfunc_2 = valfunc_2_update

        iter += 1
    end
end




# Construct aggregate transition probabilities
function compute_Q_probs(prim::Primitives, results::Results)
    @unpack ω, ω_length, q_grid = prim
    @unpack x_polfunc_1, x_polfunc_2 = results

    Q_prob = zeros(ω_length, ω_length)
    for i_ω=1:ω_length, i_ω_p=1:ω_length
        i_1 = findfirst(x->x==ω[i_ω][1], q_grid)
        i_2 = findfirst(x->x==ω[i_ω][2], q_grid)

        i_p_1 = findfirst(x->x==ω[i_ω_p][1], q_grid)
        i_p_2 = findfirst(x->x==ω[i_ω_p][2], q_grid)

        Q_prob[i_ω, i_ω_p] = (
            pr_capacity_transition(q_grid[i_1], q_grid[i_p_1], x_polfunc_1[i_1, i_2], prim) * pr_capacity_transition(q_grid[i_2], q_grid[i_p_2], x_polfunc_2[i_1, i_2], prim)
        )
    end

    results.Q_prob = Q_prob
end

# Solve model
function solve_model()
    # Initialize model primitives & struct to hold results
    prim, results = initialize()

    # Find static Cournot quantities & profits for all states
    println("Computing Cournot equilibria for all industry states . . .")
    solve_cournot(prim, results)
    println("Done.")

    # Solve firms' dynamic problems
    println("Solving firms' dynamic problems . . .")
    iterate_bellman(prim, results; maxiter=1_000)
    println("Done.")

    # Compute industry transition probabilities
    println("Computing industry transition probabilities . . .")
    compute_Q_probs(prim, results)
    println("Done.")

    return prim, results
end

# Simulate industry evolution, starting from a specific state ω
# Referred to John Higgins' Github code for this part.
# INPUTS:
#   starting state ω
#   number of periods T
#   number of replications reps
function industry_path_sim(ω_start::Tuple{Float64, Float64}, T::Int64, reps::Int64)
    @unpack ω, ω_length = prim
    @unpack Q_prob = results

    Random.seed!(1234)
    dist = Uniform(0, 1)

    ω_paths = Int64.(zeros(T, reps))

    for r=1:reps
        ω_paths[1, r] = findfirst(x->x==ω_start, ω)
        for t=2:T
            i_ω = ω_paths[t - 1, r]
            
            # To determine the next state ω', starting from ω:
            # Draw a shock ~ Unif[0,1]
            # Next state = smallest ω' s.t. F(ω'|ω) > shock,
            #   where F is the conditional cdf of ω' conditional on ω
            shock = rand(dist)
            i_ω_p = findfirst( cumsum(Q_prob[i_ω, :]) / sum(Q_prob[i_ω, :]) .> shock )

            ω_paths[t, r] = i_ω_p
        end
    end

    results.ω_paths_sim = ω_paths
end
