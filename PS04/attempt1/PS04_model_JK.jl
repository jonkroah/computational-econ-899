cd("C:\\Users\\jgkro\\Documents\\GitHub\\computational-econ-899\\PS04")

using Parameters
using DelimitedFiles
using Plots

@with_kw struct Input
    # Production parameters
    δ::Float64 = 0.06 # depreciation rate
    α::Float64 = 0.36 # capital share
    
    # HH life cycle parameters
    N::Int64 = 66 # number of periods agents live for
    n::Float64 = 0.011 # population growth rate
    Jr::Int64 = 46 # retirement age
    eta = open(readdlm,"ef.txt")
    
    # HH utility function parameters
    γ::Float64 = 0.42 # weight on consumption
    σ::Float64 = 2.0 # coefficient of relative risk aversion
    β::Float64 = 0.97 # discount factor
    
    # HH productivity Markov chain
    z::Array{Float64,1} = [3.0, 0.5] # high/low idiosyncratic productivity
    z_length::Int64 = length(z)
    Π::Array{Float64,2} = [0.9261 0.07389; 0.0189 0.9811] # random productivity persistence probabilities
    Π_ergodic::Array{Float64} = [0.2037, 0.7963] # stationary distribution of the Markov chain z
    
    # Asset grid
    A_min::Float64 = 0.0
    A_max::Float64 = 40.0
    A::Array{Float64,1} = [A_min, A_max] # space of asset holdings -- can't find this in the PS?
    a_length::Int64 = 400 # asset grid length, count
    a_grid::Array{Float64,1} = range(start=A[1],length=a_length,stop=A[2]) # asset grid

    # Social Security tax
    θ::Float64 = 0.11 # proportional labor income tax

    # Initial guesses for prices
    w_initial::Float64 = 1.05 # wage
    r_initial::Float64 = 0.05 # interest rate
    b_initial::Float64 = 0.2 # social security benefit
end

mutable struct Output_SS
    valfunc::Array{Float64,3} # value function, assets x age
    polfunc::Array{Float64,3} # policy function (capital/savings)
    labfunc::Array{Float64,3} # policy function (labor choice)
    
    F::Array{Float64,3} # steady-state distribution of agents over age, productivity, assets
    μ_age::Array{Float64} # relative size of age cohorts
    
    K::Float64 # steady-state aggregate capital
    L::Float64 # steady-state aggregate labor
    r::Float64 # steady-state interest rate
    w::Float64 # steady-state wage
    b::Float64 # steady-state pension benefit
end

function Initialize_SS(input::Input)
    @unpack a_length, N, Jr = input
    
    valfunc = zeros(a_length, N, 2)
    polfunc = zeros(a_length, N, 2)
    labfunc = zeros(a_length, Jr-1, 2)
    
    # distribution over assets, age, states
    F = zeros(a_length, N, 2)

    # relative size of age cohorts
    μ_age = zeros(N)
    
    # initial guesses for steady-state capital, labor, and prices/benefits
    K = 1
    L = 1

    r = input.r_initial
    w = input.w_initial
    b = input.b_initial
    
    return Output_SS(valfunc, polfunc, labfunc, F, μ_age, K, L, r, w, b)
end

# Retiree's utility function: no labor
function ur(c::Float64, γ::Float64, σ::Float64)
    if c>0
        return (c ^ ((1 - σ) * γ)) / (1 - σ)
    else
        return -Inf
    end
end

# Worker's utility function
function uw(c::Float64, l::Float64, γ::Float64, σ::Float64)
    if c>0
        return (((c ^ γ) * (1 - l)^(1 - γ)) ^ (1 - σ)) / (1 - σ)
    else
        return -Inf
    end
end

function solve_HH_problem(input::Input, output_SS::Output_SS)
    # Solve HH problem backwards from N (last period) to 1 (birth)
    # GOAL: for each a, j, z, solve the HH problem. Return an (a_length)x(N-Jr)x(2) matrix
    @unpack N, Jr, a_length, a_grid, eta, θ, γ, σ, β, Π, z_length, z = input
    @unpack r, w, b = output_SS # current guesses for prices

    # temporary matrix to store results; will return this
    valfunc_temp = zeros(a_length, N, z_length)
    polfunc_temp = zeros(a_length, N, z_length)
    labfunc_temp = zeros(a_length, Jr-1, z_length)

    # Retiree's problem
    for j=N:-1:Jr
        for i=1:a_length
            #oldest cohort: will die next period, so no savings
            if j==N
                max_val = ur((1 + r) * a_grid[i] + b, γ, σ)
                valfunc_temp[i, j, :] .= max_val
            else
                # continuation value: use value function for age j+1 (solved for in previous iteration of the loop)
                v_next = valfunc_temp[:, j + 1, 1]

                # calculate lifetime value at all possible levels of a' in the a_grid
                vals = ur.((1 + r) * a_grid[i] + b .- a_grid, γ, σ) + β * v_next

                # find the a' that yields the highest value
                max_index = findmax(vals)[2]
                
                # store corresponding value + a'
                valfunc_temp[i, j, :] .= vals[max_index]
                polfunc_temp[i, j, :] .= a_grid[max_index]
            end
        end
    end

    # Worker's problem
    for j=(Jr-1):-1:1
        for i_z=1:z_length
            for i=1:a_length
                # EXPECTED continuation value
                v_next = Π[i_z, 1] .* valfunc_temp[:, j + 1, 1] .+ Π[i_z, 2] .* valfunc_temp[:, j + 1, 2]

                # efficient hours worked
                e = eta[j] * z[i_z]

                # compute all possible labor choices (pinned down by choice of a')
                l = ((γ * (1 - θ) * e * w) .- ((1 - γ) * ((1 + r) * a_grid[i] .- a_grid))) ./ ((1 - θ) * w * e) # labor
                l = min.(1.0, max.(0.0, l))

                # compute all possible consumption choices (pinned down by a' and l(a'))
                c = w * (1 - θ) * e .* l .+ (1 + r) * a_grid[i] .- a_grid

                # compute lifetime values for each choice of a'
                vals = uw.(c, l, γ, σ) .+ β .* v_next

                # find the a' that yields the highest value
                max_index = findmax(vals)[2]

                # store corresponding value, savings a', and labor l
                valfunc_temp[i, j, i_z] = vals[max_index]
                polfunc_temp[i, j, i_z] = a_grid[max_index]
                labfunc_temp[i, j, i_z] = l[max_index]
            end
        end
    end

    return valfunc_temp, polfunc_temp, labfunc_temp
end

# Compute the steady-state distribution of agents over age (j), productivity (z), asset holdings (a): F_j(z, a)
function distribution(input::Input, output::Output_SS)
    @unpack N, Π, a_length, a_grid, n, z_length, Π_ergodic = input
    @unpack polfunc = output

    # Step 1: find relative sizes of each cohort of each age j
    μ_age = ones(N)
    for j=2:N
        μ_age[j] = μ_age[j-1] / (1+n)
    end
    μ_age = μ_age / sum(μ_age)
    output.μ_age = μ_age

    # Step 2: compute wealth distribution for each age cohort, using policy rules + distribution of prev cohort
    # age j=1: 
    @assert a_grid[1] == 0.0
    F = zeros(a_length, N, z_length)
    F[1, 1, 1] = μ_age[1] * Π_ergodic[1]
    F[1, 1, 2] = μ_age[1] * Π_ergodic[2]

    # age j>1: use decision rules + prev cohort's distribution of assets
    for j=2:N, i=1:a_length, s=1:z_length
        # Mathematically, this computes (for each age j, asset level (today) i, state (today) s):
        # μ_age[j] * { sum over i', s':  F[i', j-1, s'] * Indicator(polfunc[i', s', j-1] == a_grid[i]) * Π[s', s]  }
        
        temp = F[:, j - 1, :] .* (polfunc[:, j - 1, :] .== a_grid[i])
        #     # 1000x2 matrix
        #     # Each entry (i',s') is the mass of people at age (j-1) who had assets a_grid[i'] & were at state z[s'] last period,
        #     # and optimally chose a_grid[i] for the next period
        #     # i.e., temp[i', s'] = F[i', j-1, s'] * Indicator(polfunc[i', s', j-1] == a_grid[i])

        F[i, j, s] = 1/(1+n) * sum(temp' .* Π[:, s])
        #     # Multiply each term in the prev matrix by the fraction who transition from state s' (at age j-1) to state s (age j)
        #     # and then sum over s'. Finally, multiply this by the size of age cohort j
    end

    return F
end

# Update current guess for steady-state aggregate labor and capital K_SS and L_SS
function K_L_update(input::Input, output::Output_SS)
    @unpack θ, Jr, N, a_grid, a_length, z_length, eta, z = input

    # compute new guess for steady-state capital
    K_next = sum(output.F .* a_grid)

    # e[z, j]: earnings in state z at age j=1, ..., (Jr - 1) (2 x 45 matrix)
    e = eta' .* z

    # compute new guess for steady-state labor
    sum1 = sum(output.labfunc[:, :, 1] .* e[1, :]' .* output.F[:, 1:(Jr-1), 1]) # dot product for state z=1
    sum2 = sum(output.labfunc[:, :, 2] .* e[2, :]' .* output.F[:, 1:(Jr-1), 2]) # dot product for state z=2
    L_next = sum1 + sum2

    return K_next, L_next
end

# Repeatedly update guesses for K and L until convergence, then compute SS pension benefit + wages
function solve_steady_state(input::Input, output::Output_SS; tol::Float64=1e-3, maxiter::Int64=100)
    @unpack α, Jr, N, θ = input
    
    err = 1000.0

    counter = 0

    while err > tol && counter < maxiter
        # if mod(counter,5) == 0
        #     println("***** Iteration $counter:")
        # end
        
        # (1) Solve retiree and worker problems with current prices
        #println("Solving HH problem")
        output.valfunc, output.polfunc, output.labfunc = solve_HH_problem(input, output)
        
        # (2) Compute distribution of assets and productivity states across ages
        #println("Solving for stationary distribution F")
        output.F = distribution(input, output)
        
        # (3) Compute aggregate K and L implied by results from (1) and (2)
        #println("Updating capital and labor guesses")
        K_next, L_next = K_L_update(input, output)

        # Calculate difference b/w prev and current guesses for SS K and L
        err = max(abs(K_next - output.K), abs(L_next - output.L))
        if mod(counter,1) == 0
            println("Error at iteration $counter: " * string(err))
        end
        
        # Update guesses for SS K and L
        output.K = 0.3 * K_next + 0.7 * output.K
        output.L = 0.3 * L_next + 0.7 * output.L

        # Update prices based on updated K and L guesses
        output.w = (1-α) * (output.K^α) * (output.L^(-α)) # wage = MPL
        output.r = α * (output.K^(α - 1)) * (output.L^(1 - α)) # interest = MPK
        output.b = (θ * output.w * output.L) / sum(output.μ_age[Jr:N])

        counter += 1
    end

    if counter == maxiter
        println("Reached max number of iterations ($maxiter) for K, L convergence.")
    else
        println("Guesses for K_SS and L_SS converged in $counter iterations!")
    end

    # # Calculate welfare
    # welfare = output.valfunc .* output.F
    # welfare = sum(welfare[isfinite.(welfare)])
    # output.welfare = welfare
end

#----------------------
socsec_inp = Input()
socsec_out = Initialize_SS(socsec_inp)
@time solve_steady_state(socsec_inp, socsec_out)

no_socsec_inp = Input(θ=0.0)
no_socsec_out = Initialize_SS(no_socsec_inp)
solve_steady_state(no_socsec_inp, no_socsec_out)

socsec_out.K
socsec_out.L

no_socsec_out.K
no_socsec_out.L

#----------------------


#-----------------transition stuff
#=
mutable struct Output_Transition
    valfunc_t::Array{Float64,4}
    polfunc_t::Array{Float64,4}
    labfunc_t::Array{Float64,4} # policy function for each t
    
    F_t::Array{Float64,4} # distribution of agents: assets x age x state, for each time t
    μ_age_t::Array{Float64} # relative size of age cohorts
    
    K_path::Array{Float64,1} # path of aggregate capital
    L_path::Array{Float64,1} # path of aggregate labor
    
    w_path::Array{Float64,1} # path of wages
    r_path::Array{Float64,1} # path of interest rates
    b_path::Array{Float64,1} # path of pension benefits
end
=#





