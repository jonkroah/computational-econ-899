#--------------------------------------
# ECON 899 Problem Set 6
# Hopenhayn & Rogerson (deterministic)
#--------------------------------------
using Parameters, Plots

#--------------------------------------------------------------
# (0) Structs and basic functions
#--------------------------------------------------------------
# Model primitives
@with_kw struct Primitives
    #--- HOUSEHOLD PARAMETERS
    β::Float64 = 0.8        # HH discount factor
    A::Float64 = 1 / 200    # HH disutility per unit of labor

    #--- FIRM PARAMETERS
    θ::Float64 = 0.64       # labor share in production function
    c_f::Float64 = 10.0     # fixed cost for incumbents
    c_e::Float64 = 5.0      # fixed cost for entrants
    
    # Firm productivity shocks & transition probabilities
    s_grid = [3.98e-4, 3.58, 6.82, 12.18, 18.79]
    s_length = length(s_grid)
    markov = [0.6598 0.2600 0.0416 0.0331 0.0055
            0.1997 0.7201 0.0420 0.0326 0.0056
            0.2000 0.2000 0.5555 0.0344 0.0101
            0.2000 0.2000 0.2502 0.3397 0.0101
            0.2000 0.2000 0.2500 0.3400 0.0100]
    invariant_dist = [0.37, 0.4631, 0.1102, 0.0504, 0.0063] # invariant distribution from which
                                                            # entrants draw their initial productivities
    
    #--- GRIDS FOR OPTIMIZATION
    # Policy function choices: exit (x=1) or remain (x=0)
    x_grid = [0.0, 1.0]
    x_length = length(x_grid)

    # Search grid for prices
    p_min = 0.001
    p_max = 1.0
    p_length = 500
    p_grid = range(start=p_min, stop=p_max, length=p_length)
    
    # Search grid for mass of firms
    M_min = 0.0
    M_max = 10.0
    M_length = 100
    M_grid = range(start=M_min, stop=M_max, length=M_length)
end

# Struct to hold model results
mutable struct Results
    M::Float64 # mass of firms
    p::Float64 # price of output good
    
    valfunc::Array{Float64,2} # val func for incumbent / entrant who has entered
    polfunc::Array{Float64,2} # pol func for incumbent / entrant who has entered

    μ::Array{Float64, 1} # distribution of productivity states s at each price p

    EC::Float64     # error for entry condition (want close to 0 in eqm)
    LMC::Float64    # error for labor market clearing condition (want close to 0 in eqm)
end

# Function to initialize the results w/ initial guesses/placeholder values
function initialize_results(prim::Primitives)
    M = 1.0 # initial guess for equilibrium mass of firms
    p = 0.5 # initial guess for equilibrium price
    
    valfunc = zeros(prim.p_length, prim.s_length) # initialize value function
    polfunc = zeros(prim.p_length, prim.s_length) # initialize policy function

    # Distribution of productivity states
    μ = zeros(prim.s_length)

    EC = 1000.0     # initialize entry condition
    LMC = 1000.0    # initial labor market clearing condition

    # return struct w/ initial values
    results = Results(M, p, valfunc, polfunc, μ, EC, LMC)
    return results
end

#-----------------------------------------------------------------------
# (1) Incumbent's problem / entrant's problem (conditional on entering)
#-----------------------------------------------------------------------
# Firm's optimal labor demand, given output price p and productivity state s
function labor_demand(prim::Primitives, p::Float64, s::Float64)
    @unpack θ = prim
    n_d = (θ * p * s) ^ (1 / (1 - θ))
    return n_d
end

#-----------------------------------------------------------------------
# (1.1) Bellman operator
#-----------------------------------------------------------------------
function bellman(prim::Primitives, results::Results, p::Float64)
    @unpack x_grid, x_length, s_grid, s_length, p_grid, p_length, θ, c_f, β, markov = prim

    valfunc_next = zeros(p_length, s_length)
    polfunc_next = zeros(p_length, s_length)
    
    valfunc_tomorrow = results.valfunc

    for i_p=1:p_length      # loop over every possible price p
        for i_s=1:s_length  # loop over every possible state s
            p = p_grid[i_p]
            s = s_grid[i_s]

            # Firm's optimal labor demand, given (p, s)
            n = labor_demand(prim, p, s)

            # Continuation value
            continuation =  markov[i_s, 1] * valfunc_tomorrow[i_p, 1] +
                            markov[i_s, 2] * valfunc_tomorrow[i_p, 2] + 
                            markov[i_s, 3] * valfunc_tomorrow[i_p, 3] +
                            markov[i_s, 4] * valfunc_tomorrow[i_p, 4]
            
            # Calculate value for all choices of x' ∈ {1, 0}
            vals = (p * s * n^θ) .- n .- p * c_f .+ β * (1 .- x_grid) * continuation
            
            valfunc_next[i_p, i_s] = findmax(vals)[1]
            polfunc_next[i_p, i_s] = x_grid[findmax(vals)[2]]
        end
    end
    
    return valfunc_next, polfunc_next
end

#-----------------------------------------------------------------------
# (1.2) Value function iteration
#-----------------------------------------------------------------------
function valfunc_iteration(prim::Primitives, results::Results; tol::Float64=1e-4, maxiter::Int64=1000)
    error = 1e6
    counter = 0
    while error > tol && counter < maxiter
        valfunc_next, polfunc_next = bellman(prim, results)

        error = maximum(abs.(valfunc_next .- results.valfunc))

        results.valfunc = valfunc_next
        results.polfunc = polfunc_next
        counter += 1

        # println("Value function iteration $counter: error = $error")
    end
    if counter == maxiter
        println("VFI reached max number of iterations ($maxiter). Error: $error, tolerance: $tol.")
    else
        println("VFI converged in $counter iterations! Error: $error, tolerance: $tol.")
    end
end

#-----------------------------------------------------------------------
# (2) Free entry condition --> pin down equilibrium price
#-----------------------------------------------------------------------
function find_eqm_price(prim::Primitives, results::Results; tol=0.01)
    @unpack p_grid, p_length, invariant_dist, s_length, c_e = prim
    @unpack valfunc = results

    # Compute the entry condition at every possible price
    EC = zeros(p_length)
    for i_p=1:p_length
        p = p_grid[i_p]
        EC[i_p] = (valfunc * invariant_dist)[i_p] / p - c_e
    end

    # Find the price that makes the entry condition closest to 0
    EC_eqm = findmin(abs.(EC))[1]
    p_eqm = p_grid[findmin(abs.(EC))[2]]

    @assert EC_eqm < tol "Free entry condition is not close enough to 0 (tol=$tol)."
    
    return p_eqm, EC_eqm
end

#-----------------INITIALIZE STUFF -- DELETE LATER
prim = Primitives(p_length=1000)
results = initialize_results(prim)

# Solve firm's dynamic problem
valfunc_iteration(prim, results)

# Use free entry condition to find equilibrium price
results.p, results.EC = find_eqm_price(prim, results; tol=0.01)

# Solve for stationary distribution μ
@unpack p_grid, markov, invariant_dist = prim
@unpack p, polfunc = results

i_sp = 1

# index of equilibrium price
i_p = indexin(p, p_grid)

# 1 - X(s, p*)
1 .- polfunc[i_p, :]






# Graph firm value & policy functions
@unpack p_grid, s_grid = prim
@unpack valfunc, polfunc = results

plot(p_grid, valfunc, labels=["s=3.98e-4" "s=3.58" "s=6.81" "s=12.18" "s=18.79"], legend=:topleft, linewidth=2)
plot(p_grid, polfunc, labels=["s=3.98e-4" "s=3.58" "s=6.81" "s=12.18" "s=18.79"], legend=:bottomleft, linewidth=2)