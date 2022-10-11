#------------------------------------------------
# AUTHOR: Jon Kroah
# DATE CREATED: 08 Oct 2022
# Based on reference code from Phil Coyle.
# Also referenced code from Katherine Kwok.
#------------------------------------------------
using Plots, Parameters, Random, Distributions, Interpolations, Optim, DataFrames, GLM

#------------------------------------------------------------------------------
# (0) Structs to hold model primitives, grids, transition probabilities,
#       HH prob solutions, and coefs for aggregate capital transition function.
#     Functions to perform basic calculations (e.g., MPK/MPL).
#------------------------------------------------------------------------------
#--- Model parameters and grids for optimization
@with_kw struct Primitives
    # HH parameters
    β::Float64 = 0.99 # HH discount factor
    e_bar::Float64 = 0.3271 # effective units of labor supplied (if employed)

    # Production/capital law of motion
    α::Float64 = 0.36 # capital share for Cobb-Douglas
    δ::Float64 = 0.025 # depreciation
    
    # Idiosyncratic shocks (ε)
    ε_grid::Array{Float64} = [1, 0] # idiosyncratic employment opportunities
    ε_length::Int64 = length(ε_grid)

    # Aggregate shocks (z)
    z_g::Float64 = 1.01 # good state
    z_b::Float64 = 0.99 # bad state
    z_grid::Array{Float64} = [z_g, z_b]
    z_length::Int64 = length(z_grid)

    # Grid for HH savings (k)
    k_lb::Float64 = 0.001
    k_ub::Float64 = 20.0
    k_length::Int64 = 21
    k_grid::Array{Float64, 1} = range(start = k_lb, stop = k_ub, length = k_length)

    # Grid for aggregate capital stock (K)
    K_lb::Float64 = 10.0
    K_ub::Float64 = 15.0
    K_length::Int64 = 11
    K_grid::Array{Float64, 1} = range(start = K_lb, stop = K_ub, length = K_length)

    # number of households and time periods for simulation
    N::Int64 = 5000 # number of HHs to draw idiosyncratic ε shocks for
    T::Int64 = 11000 # number of time periods to draw agg z shocks for
    T_discard::Int64 = 1000 # number of time periods to discard
end

#--- Construct & store transition probabilities for aggregate and idiosyncratic shocks
@with_kw struct Shocks
    #parameters of transition matrix:
    u_dur_g::Float64 = 1.5 # Unemp Duration in good agg state
    u_dur_b::Float64 = 2.5 # Unemp Duration in bad agg state
    u_g::Float64 = 0.04 # Fraction Unemp in good agg state
    u_b::Float64 = 0.1 # Fraction Unemp in bad agg state
    dur_g::Float64 = 8.0 # Duration in good agg state
    dur_b::Float64 = 8.0 # Duration in bad agg state

    #transition probabilities for aggregate states
    pgg::Float64 = (dur_g - 1.0) / dur_g
    pgb::Float64 = 1.0 - (dur_b - 1.0) / dur_b
    pbg::Float64 = 1.0 - (dur_g - 1.0) / dur_g
    pbb::Float64 = (dur_b - 1.0) / dur_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (u_dur_g - 1.0) / u_dur_g
    pbb00::Float64 = (u_dur_b - 1.0) / u_dur_b
    pbg00::Float64 = 1.25 * pbb00
    pgb00::Float64 = 0.75 * pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g * pgg00) / (1.0 - u_g)
    pbb01::Float64 = (u_b - u_b * pbb00) / (1.0 - u_b)
    pbg01::Float64 = (u_b - u_g * pbg00) / (1.0 - u_g)
    pgb01::Float64 = (u_g - u_b * pgb00) / (1.0 - u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (u_dur_g - 1.0) / u_dur_g
    pbb10::Float64 = 1.0 - (u_dur_b - 1.0) / u_dur_b
    pbg10::Float64 = 1.0 - 1.25 * pbb00
    pgb10::Float64 = 1.0 - 0.75 * pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g * pgg00) / (1.0 - u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b * pbb00) / (1.0 - u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g * pbg00) / (1.0 - u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b * pgb00) / (1.0 - u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    markov::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                                pbg*Mbg pbb*Mbb]
end

#--- Main results
mutable struct Output
    # Solution to the HH problem: value & policy functions
    #   Note, these are 4-dimensional: depend on (k, ε, K, z)
    valfunc::Array{Float64, 4}
    polfunc::Array{Float64, 4}

    # Coefficients for the law of motion in the GOOD state (z_g)
    a0::Float64 # intercept
    a1::Float64 # slope

    # Coefficients for the law of motion in the BAD agg state (z_b)
    b0::Float64 # intercept
    b1::Float64 # slope

    # R² for each regression
    R2_a::Float64
    R2_b::Float64
end

#--- Initialize output struct with initial guesses/placeholders
function initialize_output(prim::Primitives)
    @unpack k_length, ε_length, K_length, z_length = prim

    # Initialize value & policy functions: start with zeros
    valfunc = zeros(k_length, ε_length, K_length, z_length)
    polfunc = zeros(k_length, ε_length, K_length, z_length)

    # Coefficients for the law of motion in the GOOD state (z_g)
    a0 = 0.0
    a1 = 1.0

    # Coefficients for the law of motion in the BAD state (z_g)
    b0 = 0.0
    b1 = 0.5

    # R² for regression
    R2_a = 0.0
    R2_b = 0.0

    output = Output(valfunc, polfunc, a0, a1, b0, b1, R2_a, R2_b)
    return output
end

#--- Calculate marginal products of capital & labor (i.e., prices)
function MPK_MPL(prim::Primitives, K::Float64, L::Float64, z::Float64)
    @unpack α = prim
    MPK = α * z * (K / L)^(α - 1)
    MPL = (1 - α) * z * (K / L)^α
    return MPK, MPL
end

#--------------------------------------------------------------------------
# (2) Draw panel of aggregate and HH-level shocks
#--------------------------------------------------------------------------
function draw_shocks(shocks::Shocks, N::Int64, T::Int64)
    @unpack pgg, pgb, pbb, pbg, Mgg, Mgb, Mbb, Mbg = shocks
    
    Random.seed!(10092022) # IMPORTANT: for replication
    
    dist = Uniform(0, 1)

    z_state = fill(0, T)      # series of T aggregate states
    ε_state = fill(0, N, T)   # N HHs, each w/ a series of T idiosyncratic shocks

    z_state[1] = 1          # for t=1, start economy in the good state (z=z_g)
    ε_state[:, 1] .= 1      # for t=1, start all HHs employed (ε=1)

    # For each time period, draw aggregate shock.
    # Then draw idiosyncratic shocks for all HHs that period.
    for t=2:T
        # Aggregate shocks for time t:
        #   Draw random number ~ Unif[0,1].
        #   If it's < 0.875 (= pgg = pbb), then stay at same state as prev period.
        #   O/w, move to different state.
        z_shock = rand(dist)
        if z_state[t - 1] == 1 && z_shock <= pgg # good state last pd.
            z_state[t] = 1 # stay at good state
        elseif z_state[t - 1] == 1 && z_shock > pgg
            z_state[t] = 2 # move to bad state
        elseif z_state[t - 1] == 2 && z_shock <= pbb # bad state last pd.
            z_state[t] = 2 # stay at bad state
        elseif z_state[t - 1] == 2 && z_shock > pbb
            z_state[t] = 1 # move to good state
        end

        # Idiosyncratic shocks for each HH in time t:
        #   Draw random number ~ Unif[0, 1]
        #   Aggregate states in t, t-1 determine the ε transition probabilities.
        #   If idio state was ε in t-1, and random number <= p_εε, then HH stays
        #       at ε; o/w, move HH to other state ε'.
        for n=1:N
            ε_shock = rand(dist)

            # get ε transition probabilities, based on z today & yesterday
            p11, p00 = 0.0, 0.0
            if z_state[t-1] == 1 && z_state[t] == 1 # g -> g
                p11 = Mgg[1,1]
                p00 = Mgg[2,2]
            elseif z_state[t-1] == 1 && z_state[t] == 2 # g -> b
                p11 = Mgb[1,1]
                p00 = Mgb[2,2]
            elseif z_state[t-1] == 2 && z_state[t] == 1 # b -> g
                p11 = Mbg[1,1]
                p00 = Mbg[2,2]
            elseif z_state[t-1] == 2 && z_state[t] == 2 # b -> b
                p11 = Mbb[1,1]
                p00 = Mbb[2,2]
            end
            @assert p00 > 0.0 && p11 > 0.0 # make sure p00, p11 updated
            
            # move HHs based on random number drawn vs. transition probs
            if ε_state[n, t-1] == 1 && ε_shock <= p11 # employed last pd
                ε_state[n, t] = 1 # stay empl.
            elseif ε_state[n, t-1] == 1 && ε_shock > p11
                ε_state[n, t] = 2 # move to unempl.
            elseif ε_state[n, t-1] == 2 && ε_shock <= p00 # unemployed last pd
                ε_state[n, t] = 2 # stay unempl.
            elseif ε_state[n, t-1] == 2 && ε_shock > p00
                ε_state[n, t] = 1 # move to empl.
            end
        end
    end
    return ε_state, z_state
end

#--------------------------------------------------------------------------
# (3) Solve HH problem
# (3.1) Bellman operator (v_today --> v_next)
# (3.2) Value function iteration
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# (3.1) Bellman operator
#--------------------------------------------------------------------------
function bellman(prim::Primitives, shocks::Shocks, output::Output)
    @unpack k_grid, k_length, K_grid, K_length, ε_length, ε_grid, z_length, z_grid, β, e_bar, δ = prim
    @unpack u_g, u_b, markov = shocks
    @unpack a0, a1, b0, b1, valfunc, polfunc = output

    valfunc_next = zeros(k_length, ε_length, K_length, z_length)
    polfunc_next = zeros(k_length, ε_length, K_length, z_length)

    # Interpolate grids
    k_interp        = interpolate(k_grid, BSpline(Linear()))
    valfunc_interp  = interpolate(valfunc, BSpline(Linear()))

    # Need to solve the HH problem
    #   For each aggregate state z
    #       For each level of aggregate capital K
    #           For each idiosyncratic employment state
    #               For each current level of asset holdings

    # For each aggregate state z today:
    for (i_z, z_today) in enumerate(z_grid)
    # i_z = 1
    # z_today = z_grid[i_z]
        
        # Calculate aggregate labor supply (only depends on aggregate state today)
        if i_z == 1 # aggregate state is good
            L_today = e_bar * (1 - u_g)
        elseif i_z == 2 # aggregate state is bad
            L_today = e_bar * (1 - u_b)
        end

        # For each level of aggregate capital K today:
        for (i_K, K_today) in enumerate(K_grid)
        # i_K = 6
        # K_today = K_grid[i_K]

            # Calculate prices
            r_today, w_today = MPK_MPL(prim, K_today, L_today, z_today)

            # Calculate avg K tomorrow, based on conjectured law of motion for avg K
            if i_z == 1 # aggregate state is good
                K_tomorrow = a0 + a1 * log(K_today)
            elseif i_z == 2 # aggregate state is bad
                K_tomorrow = b0 + b1 * log(K_today)
            end
            K_tomorrow = exp(K_tomorrow)

            # Get index of K_tomorrow in INTERPOLATED K grid
            #   This will point to location in INTERPOLATED value function
            i_Kp = get_index(K_tomorrow, K_grid)

            # For each idiosyncratic state ε today:
            for (i_ε, ε_today) in enumerate(ε_grid)
            # i_ε = 1
            # ε_today = ε_grid[i_ε]
                
                # Get the row of the 4x4 Markov transition matrix
                #   corresponding to the states (ε, z) today:
                #       (i_ε, i_z) = (1, 1) --> row = 1
                #       (i_ε, i_z) = (2, 1) --> row = 2
                #       (i_ε, i_z) = (1, 2) --> row = 3
                #       (i_ε, i_z) = (2, 2) --> row = 4
                # Not sure whether this calc works if either
                #   dimension of the matrix is >2, but keeping
                #   this code for now.
                row = i_ε + ε_length * (i_z - 1)
                
                # For each level of HH capital holdings k today:
                for (i_k, k_today) in enumerate(k_grid)
                # i_k = 10
                # k_today = k_grid[i_k]

                    # Budget today: capital + labor income
                    budget_today = (1 + r_today - δ) * k_today + w_today * ε_today * e_bar

                    # Continuation value
                    # Note that this defines a FUNCTION, mapping indices of the interpolated k-grid (i_kp)
                    #   to a (vector-valued) value function V(k, ε, K, z).
                    # We will pass this function into an optimization function below to get the optimal
                    #   V and associated optimizer (policy fn) k'.
                    # Note on column indices for Markov transition matrix vs. indices for valfunc:
                    #       (1,1) = column 1, (2,1) = column 2, (1,2) = column 3, (2,2) = column 4
                    v_tomorrow(i_kp) =  markov[row, 1] * valfunc_interp(i_kp, 1, i_Kp, 1) +
                                        markov[row, 2] * valfunc_interp(i_kp, 2, i_Kp, 1) +
                                        markov[row, 3] * valfunc_interp(i_kp, 1, i_Kp, 2) +
                                        markov[row, 4] * valfunc_interp(i_kp, 2, i_Kp, 2)
                    
                    # Total present value
                    # FUNCTION mapping indices of the interpolated k-grid (i_kp) to
                    vals_today(i_kp) = log(budget_today - k_interp(i_kp)) + β * v_tomorrow(i_kp)

                    # Translate into minimization problem
                    objective(i_kp) = -vals_today(i_kp)
                    
                    # Define lower & upper bounds of the search interval
                    lower = 1.0                             # k' >= 1.0
                    upper = get_index(budget_today, k_grid) # k' <= budget_today

                    # Solve optimization problem
                    opt = optimize(objective, lower, upper)

                    # Note that the optimizer is an INDEX --> get corresponding value of k'
                    k_tomorrow = k_interp(opt.minimizer)

                    # And take the negative of the optimum
                    v_today = -opt.minimum

                    # Store results
                    polfunc_next[i_k, i_ε, i_K, i_z] = k_tomorrow
                    valfunc_next[i_k, i_ε, i_K, i_z] = v_today
                end # end loop over k
            end # end loop over ε
        end # end loop over K
    end # end loop over z

    return valfunc_next, polfunc_next
end

#---------------------------------------------
# Find a given value in an INTERPOLATED grid
#   Note, this will return non-integer indices
#       which will get plugged into INTERPOLATED grids
#   Example code to see how this works:
#       get_index(11.24, K_grid) # this returns something like 3.48
#       temp = interpolate(K_grid, BSpline(Linear())) # interpolate the grid
#       temp[get_index(11.24, K_grid)] # this returns 11.24
function get_index(val::Float64, grid::Array{Float64,1})
    n = length(grid)
    index = 0
    if val <= grid[1]
        index = 1
    elseif val >= grid[n]
        index = n
    else
        index_upper = findfirst(x->x>val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]
        
        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    return index
end

#--------------------------------------------------------------------------
# (3.2) Value function iteration
#--------------------------------------------------------------------------
function vf_iteration(prim, shocks, output; tol::Float64=1e-5, maxiter::Int64=10000)
    error = 1e6
    counter = 0
    while error > tol && counter < maxiter
        valfunc_next, polfunc_next = bellman(prim, shocks, output)
        
        error = maximum(abs.(valfunc_next - output.valfunc))
        
        output.valfunc = valfunc_next
        output.polfunc = polfunc_next
        counter += 1

        # println("Iteration $counter: Error = $error")
    end

    if counter == maxiter
        println("VFI reached max number of iterations ($maxiter). Error: $error, tolerance: $tol.")
    else
        println("VFI converged in $counter iterations! Error: $error, tolerance: $tol.")
    end
end

#--------------------------------------------------------------------------
# (4) Plug simulated shocks into HH solutions to get simulate panel of 
#       HH savings decisions. Aggregate these up to get a time series of 
#       average capital.
#--------------------------------------------------------------------------
function savings_panel(prim::Primitives, output::Output, ε_state::Array{Int64}, z_state::Array{Int64}, N::Int64, T::Int64, T_discard::Int64)
    @unpack k_grid, ε_grid, K_grid, z_grid = prim

    # Check that dimension of shock matrices matches the N,T specified
    @assert size(ε_state) == (N, T) "savings_panel() ERROR: ε_state is not N x T."
    @assert size(z_state) == (T,) "savings_panel() ERROR: z_state is not of length T."

    # Interpolate policy function
    polfunc_interp = interpolate(output.polfunc, BSpline(Linear()))

    # Will fill these in:
    #   Panel of savings decisions for N HHs in T periods
    #   Time series of aggregate capital
    savings_panel   = zeros(N, T)
    K_series        = zeros(T)

    # For each t and each HH i, compute their savings decision
    #   Then average these to get aggregate savings for t
    #   Then carry that forward to solve for t+1, and so on.
    for t=1:T
        for i=1:N
            if t==1 # First period: Economy starts w/ steady-state level of K=11.55 (see handout). Every HH endowed w/ this amt.
                K = 11.55
                k = K
            elseif t > 1 # t>1: Economy enters t w/ K from end of last period; HHs enter t w/ savings from end of last period
                K = K_series[t - 1]
                k = savings_panel[i, t - 1]
            end

            # Get (interpolated) grid indices of current aggregate capital & HH capital
            i_K = get_index(K, K_grid)
            i_k = get_index(k, k_grid)

            # Get index of HH's current empl state
            i_ε = ε_state[i,t]

            # Get index of current agg state
            i_z = z_state[t]
            
            # Compute savings for HH i at time t & store in panel
            savings_panel[i, t] = polfunc_interp[i_k, i_ε, i_K, i_z]
        end

        # Calculate aggregate capital: average of all HH savings choices
        K_series[t] = mean(savings_panel[:, t])
    end

    # Discard first 1000 periods from panels (to remove initial condition dependence)
    savings_panel   = savings_panel[:, (T_discard + 1):T]
    K_series        = K_series[(T_discard + 1):T]
    z_state_trimmed = z_state[(T_discard + 1):T]

    return savings_panel, K_series, z_state_trimmed
end

#-------------------------------------------------------------------------------------
# (5) Run regression w/ simulated time series for average K, and use results
#       to update conjectured law of motion for average K.
#-------------------------------------------------------------------------------------
function update_law_of_motion(z_state::Array{Int64}, K_series::Array{Float64})
    @assert length(z_state) == length(K_series) "update_law_of_motion() ERROR: z_state and K_series are different lengths."

    # Get # time periods
    T = length(z_state)
    
    # Build regression dataset: want to regress K_{t+1} on K_{t}, separately for each value of z_{t}
    df = DataFrame(t=1:T, z=z_state, log_K=log.(K_series))

    # Add K_{t+1}
    df[!, :log_K_next] .= df.log_K
    for t=1:T-1
        df.log_K_next[t] = df.log_K[t + 1]
    end
    df = df[1:T-1, :] # Drop last row (no t+1 for that row)

    # Subset data based on value of z_t
    df1 = filter(:z => z -> z==1, df)
    df2 = filter(:z => z -> z==2, df)

    # Run autoregression separately for z=1 (good state) and z=2 (bad state) samples
    ols1 = lm(@formula(log_K_next ~ log_K), df1)
    ols2 = lm(@formula(log_K_next ~ log_K), df2)
    
    # Store & return results
    a0_next, a1_next = coef(ols1)       # parameters for good state (z=1)
    b0_next, b1_next = coef(ols2)       # parameters for bad state (z=2)
    R2_a_next = r2(ols1)
    R2_b_next = r2(ols2)

    return a0_next, a1_next, b0_next, b1_next, R2_a_next, R2_b_next
end

#-------------------------------------------------------------------------------------
# (6) Iterate steps (3)-(5) above until the coefficients for the law of motion
#       of average K converge.
#-------------------------------------------------------------------------------------
function solve_model(prim::Primitives, shocks::Shocks, output::Output, ε_state::Array{Int64}, z_state::Array{Int64}; 
    tol::Float64=1e-4, maxiter::Int64=1000, update_factor::Float64=0.5)
    
    @unpack N, T, T_discard = prim
    
    err = 1000.0
    counter = 0
    convergence_flag = 0
    while convergence_flag == 0 && counter < maxiter
        println("**********************")
        # Solve the HH problem; get value & policy functions for every (k,ε,K,z)
        vf_iteration(prim, shocks, output)
        println("Solved HH problem")

        # Compute savings panel & time series of average capital
        HH_savings, K_series, z_state_trimmed = savings_panel(prim, output, ε_state, z_state, N, T, T_discard)
        println("Simulated savings and aggregate capital")

        a0_next, a1_next, b0_next, b1_next, R2_a_next, R2_b_next = update_law_of_motion(z_state_trimmed, K_series)
        println("Updated capital law of motion")

        # Calculate diff b/w next guess & current guess
        err = max(abs(a0_next - output.a0), abs(a1_next - output.a1), abs(b0_next - output.b0), abs(b1_next - output.b1))

        # If next guess of coefs are too far from current guess, OR the goodness-of-fit from EITHER regression is too low,
        #   then update guess of coefs (weighted avg of next & current guesses).
        #   Otherwise, store the next guess as the final results & stop the loop.
        if err >= tol || R2_a_next < 0.99 || R2_b_next < 0.99
            output.a0 = update_factor * a0_next + (1 - update_factor) * output.a0
            output.a1 = update_factor * a1_next + (1 - update_factor) * output.a1
            output.b0 = update_factor * b0_next + (1 - update_factor) * output.b0
            output.b1 = update_factor * b1_next + (1 - update_factor) * output.b1
            output.R2_a = R2_a_next
            output.R2_b = R2_a_next
        else
            convergence_flag = 1
            output.a0 = a0_next
            output.a1 = a1_next
            output.b0 = b0_next
            output.b1 = b1_next
            output.R2_a = R2_a_next
            output.R2_b = R2_a_next
        end

        counter += 1
        println("Iteration $counter: Err=$err, Tol=$tol, R²=$R2_a_next (good state), $R2_b_next (bad state)")
    end

    if counter == maxiter
        println("solve_model() reached max number of iterations ($maxiter). Error=$err, tolerance=$tol.")
    else
        println("Model converged in $counter iterations! Error=$err, tolerance=$tol.")
        println("Results:")
        println("Law of motion in GOOD state (z=1): a0 = $(output.a0), a1 = $(output.a1), R²=$(output.R2_a)")
        println("Law of motion in GOOD state (z=1): b0 = $(output.b0), b1 = $(output.b1), R²=$(output.R2_b)")
    end
end