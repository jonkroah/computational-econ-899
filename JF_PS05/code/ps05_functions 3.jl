
# Computational (2nd half), PS5: capacity game
# Authors: Jackson Crawford, Dalya Elmalt, Jon Kroah
# Referenced code from Alex von Hafften.

using Parameters, DataFrames, LatexPrint, Plots

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
    
    prim = Primitives(δ, α, β, a, b, q_grid_min, q_grid_max, q_grid_step, q_grid, q_grid_length)

    # Results
    q_star_1 = zeros(q_grid_length, q_grid_length)
    q_star_2 = zeros(q_grid_length, q_grid_length)
    
    π_star_1 = zeros(q_grid_length, q_grid_length)
    π_star_2 = zeros(q_grid_length, q_grid_length)

    valfunc_1 = zeros(q_grid_length, q_grid_length)
    valfunc_2 = zeros(q_grid_length, q_grid_length)

    x_polfunc_1 = zeros(q_grid_length, q_grid_length)
    x_polfunc_2 = zeros(q_grid_length, q_grid_length)

    results = Results(q_star_1, q_star_2, π_star_1, π_star_2, valfunc_1, valfunc_2, x_polfunc_1, x_polfunc_2)

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
function cournot_eq(q_bar_1::Float64, q_bar_2::Float64, prim::Primitives; tol::Float64=1e-12, update_factor::Float64=0.5)

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
            println("Converged to Cournot NE in $count iterations. NE strategies = ($q_star_1, $q_star_2)")
        end
    end

    return q_star_1, q_star_2
end

# Solve Cournot equilibria for all possible industry states
# INPUT: industry states (q_grid), results
# OUTPUT: none; modifies Results struct
function solve_cournot(prim::Primitives, results::Results)
    @unpack q_grid = prim
    for (i_1, q_bar_1) in enumerate(q_grid)
        for (i_2, q_bar_2) in enumerate(q_grid)
            q_star_1, q_star_2 = cournot_eq(q_bar_1, q_bar_2, prim; tol)
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

    # for (i_q_1, q_bar_1) in enumerate(q_grid), (i_q_2, q_bar_2) in enumerate(q_grid)
    for i_q_1=1:q_grid_length, i_q_2=1:q_grid_length
        q_bar_1 = q_grid[i_q_1]
        q_bar_2 = q_grid[i_q_2]
        # println("$q_bar_1, $q_bar_2")
        #---------------------------
        # FOR TESTING
        # i_q_1 = 2
        # i_q_2 = 1
        # q_bar_1 = q_grid[i_q_1]
        # q_bar_2 = q_grid[i_q_2]
        #---------------------------

        #------------------------------------------------------------------------------------
        # ---- NEW METHOD
        # Fixing x_2, find optimal x_1 (using formula in Besanko & Doraszelski)
        # Calculate formula ingredients (handling boundary cases -- referred to Alex von Hafften's code here)
        W_1_nochange    = W(q_bar_1, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, results.valfunc_1)
        if i_q_1 == 1            # capacity can't decrease if currently at the min
            W_1_down        = W_1_nochange
        else
            W_1_down        = min(W(q_bar_1 - q_grid_step, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, results.valfunc_1), W_1_nochange)
        end
        if i_q_1 == q_grid_length # capacity can't increase if currently at the max
            W_1_up          = W_1_nochange
        else
            W_1_up          = max(W(q_bar_1 + q_grid_step, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, results.valfunc_1), W_1_nochange)
        end

        x_polfunc_1_update[i_q_1, i_q_2] = max(
            0.0,
            (-1.0 + sqrt(β * α * ((1 - δ) * (W_1_up - W_1_nochange) + δ * (W_1_nochange - W_1_down)))) / α
        )

        valfunc_1_update[i_q_1, i_q_2] = π_star_1[i_q_1, i_q_2] - x_polfunc_1_update[i_q_1, i_q_2]
        for (i, q_bar_1_p) in enumerate(q_grid)
            valfunc_1_update[i_q_1, i_q_2] += β * W(q_bar_1_p, q_bar_2, results.x_polfunc_2[i_q_1, i_q_2], prim, results.valfunc_1) * pr_capacity_transition(q_bar_1, q_bar_1_p, x_polfunc_1_update[i_q_1, i_q_2], prim)
        end
        

        # Fixing x_1, find optimal x_2 (using formula in Besanko & Doraszelski)
        # NOTE: Need to transpose firm 2's value function (matrix), b/c o/w rows are opponent's state (q_bar_1) and columns are own state (q_bar_2).
        # W function relies on the rows being *own* state & cols being *opponent's* state.
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
        #------------------------------------------------------------------------------------

        #------------------------------------------------------------------------------------
        #---- OLD METHOD
        # Solve for optimal investment (x_1_star, x_2_star).
        # Firms' optimizations are interdependent, so need to solve them simultaneously
        # x_1_star = 0.0      # initial guess
        # x_2_star = 0.0      # initial guess
        # conv_flag = 0
        # while conv_flag == 0
        #     x_1_star_next = x_1_star
        #     x_2_star_next = x_2_star

        #     # Fixing x_2, find optimal x_1 (using formula in Besanko & Doraszelski)
        #     # Calculate formula ingredients (handling boundary cases -- referred to Alex von Hafften's code here)
        #     W_1_nochange    = W(q_bar_1, q_bar_2, x_2_star_next, prim, valfunc_1)
        #     if i_q_1 == 1 # capacity can't decrease if at the min
        #         W_1_down        = W_1_nochange 
        #     else
        #         W_1_down        = min(W(q_bar_1 - q_grid_step, q_bar_2, x_2_star_next, prim, valfunc_1), W_1_nochange)
        #     end
        #     if i_q_1 == q_grid_length # capacity can't increase if at the max
        #         W_1_up          = W_1_nochange
        #     else
        #         W_1_up          = max(W(q_bar_1 + q_grid_step, q_bar_2, x_2_star_next, prim, valfunc_1), W_1_nochange)
        #     end

        #     x_1_star_next = max(
        #         0.0,
        #         (-1.0 + sqrt(β * α * ((1 - δ) * (W_1_up - W_1_nochange) + δ * (W_1_nochange - W_1_down)))) / α
        #     )
        #     x_polfunc_1_update[i_q_1, i_q_2] = x_1_star_next

        #     valfunc_1_update[i_q_1, i_q_2] = π_star_1[i_q_1, i_q_2] - x_1_star_next
        #     for (i, q_bar_1_p) in enumerate(q_grid)
        #         valfunc_1_update[i_q_1, i_q_2] += β * W(q_bar_1_p, q_bar_2, x_2_star_next, prim, valfunc_1) * pr_capacity_transition(q_bar_1, q_bar_1_p, x_1_star_next, prim)
        #     end
            

        #     # Fixing x_1, find optimal x_2 (using formula in Besanko & Doraszelski)
        #     vf_2 = Float64.(results.valfunc_2')
        #     W_2_nochange    = W(q_bar_2, q_bar_1, x_1_star_next, prim, vf_2)
        #     if i_q_2 == 1 # capacity can't decrease if at the min
        #         W_2_down        = W_2_nochange
        #     else
        #         W_2_down        = min(W(q_bar_2 - q_grid_step, q_bar_1, x_1_star_next, prim, vf_2), W_2_nochange)
        #     end
        #     if i_q_2 == q_grid_length # capacity can't increase if at the max
        #         W_2_up          = W_2_nochange
        #     else
        #         W_2_up          = max(W(q_bar_2 + q_grid_step, q_bar_1, x_1_star_next, prim, vf_2), W_2_nochange)
        #     end

        #     x_2_star_next = max(
        #         0.0,
        #         (-1.0 + sqrt(β * α * ((1 - δ) * (W_2_up - W_2_nochange) + δ * (W_2_nochange - W_2_down)))) / α
        #     )
        #     x_polfunc_2_update[i_q_1, i_q_2] = x_2_star_next

        #     valfunc_2_update[i_q_1, i_q_2] = π_star_2[i_q_1, i_q_2] - x_2_star_next
        #     for (i, q_bar_2_p) in enumerate(q_grid)
        #         valfunc_2_update[i_q_1, i_q_2] += β * W(q_bar_2_p, q_bar_1, x_1_star_next, prim, vf_2) * pr_capacity_transition(q_bar_2, q_bar_2_p, x_2_star_next, prim)
        #     end
            
        #     # Stop if new guess for optimizer is close to previous one
        #     err = max(abs(x_1_star - x_1_star_next), abs(x_2_star - x_2_star_next))
        #     if err < tol
        #         conv_flag = 1
        #     end

        #     # Update guesses for optimizers
        #     x_1_star = x_1_star_next
        #     x_2_star = x_2_star_next
        # end
        #------------------------------------------------------------------------------------

        println("(q_bar_1, q_bar_2) = ($q_bar_1, $q_bar_2): x_1_star = $(x_polfunc_1_update[i_q_1, i_q_2]), x_2_star = $(x_polfunc_2_update[i_q_1, i_q_2])")
    end

    return valfunc_1_update, valfunc_2_update, x_polfunc_1_update, x_polfunc_2_update
end

# Do value function iteration
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
            println("---Value function iteration $iter: error = $err")
        end

        # Update policy functions
        results.x_polfunc_1 = x_polfunc_1_update
        results.x_polfunc_2 = x_polfunc_2_update

        results.valfunc_1 = valfunc_1_update
        results.valfunc_2 = valfunc_2_update

        iter += 1
    end
end


#-----------------------------------------------------------
prim, results = initialize()
solve_cournot(prim, results)
iterate_bellman(prim, results; maxiter=1_000)

@unpack q_grid = prim
@unpack valfunc_1, valfunc_2, x_polfunc_1, x_polfunc_2 = results

x_polfunc_1
x_polfunc_2

plot(q_grid, q_grid, valfunc_1, st=:surface)

valfunc_1
valfunc_2