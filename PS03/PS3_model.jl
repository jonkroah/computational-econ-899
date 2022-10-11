# See PS3_script.jl which uses this file
# This file includes the economic model itself as well as other functions.

# NOTE: If you get an error complaining about redefining a constant, you may need to kill and restart the Julia terminal
#cd("/Users/dalya/Documents/GitHub/compdjj/PS3")

using Parameters
using DelimitedFiles
using Plots

# Input - exogenous selections, also including grid parameters
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
    A::Array{Float64,1} = [0.0, 100.0] # space of asset holdings -- can't find this in the PS?
    a_length::Int64 = 1000 # asset grid length, count
    a_grid::Array{Float64,1} = range(start=A[1],length=a_length,stop=A[2]) # asset grid
    
    # Taxes and prices faced by HHs
    θ::Float64 = 0.11 # proportional labor income tax
    w::Float64 = 1.05 # wage
    r::Float64 = 0.05 # interest rate
    b::Float64 = 0.2 # social security benefit
end

# Output
mutable struct Output
    valfunc::Array{Float64,3} # value function, assets x age
    polfunc::Array{Float64,3} # policy function (capital/savings)
    labfunc::Array{Float64,3} # policy function (labor choice)
    F::Array{Float64,3} # steady-state distribution of agents over age, productivity, assets
    μ_age::Array{Float64} # relative size of age cohorts
    K_SS::Float64 # steady-state capital
    L_SS::Float64 # steady-state labor
    r_SS::Float64 # steady-state interest rate
    w_SS::Float64 # steady-state wage
    b_SS::Float64 # steady-state pension benefit
    welfare::Float64 # societal welfare
end

# Initialize structures
function Initialize(input::Input)
    @unpack a_length, N, Jr = input
    valfunc = zeros(a_length,N,2)
    polfunc = zeros(a_length,N,2)
    labfunc = zeros(a_length,Jr-1,2)
    
    # distribution over assets, age, states
    F = zeros(a_length, N, 2)

    # relative size of age cohorts
    μ_age = zeros(N)
    
    # initial guesses for steady-state capital, labor, and prices/benefits
    K_SS = 10
    L_SS = 1
    r_SS = input.r
    w_SS = input.w
    b_SS = input.b

    welfare = 0
    
    return Output(valfunc,polfunc,labfunc,F, μ_age, K_SS, L_SS, w_SS, r_SS, b_SS, welfare)
end

# Worker utility function
function uw(c::Float64,l::Float64,γ::Float64,σ::Float64)
    if c>0
        return (((c^γ)*(1-l)^(1-γ))^(1-σ))/(1-σ)
    else
        return -Inf
    end
end

# Retiree utility function
function ur(c::Float64,γ::Float64,σ::Float64)
    if c>0
        return (c^((1-σ)*γ))/(1-σ)
    else
        return -Inf
    end
end    

# Solve retiree problem (backwards)
function retiree(input::Input,output::Output;updating::Bool=false)
    @unpack γ, σ, a_length, a_grid, N, Jr, r, b, β = input

    if updating == true
        r = output.r_SS
        b = output.b_SS
    end

    for j = N:-1:Jr
        for i = 1:a_length # iterate over possible asset states
            max_val = -Inf
            if j == N # last period, no real choice to make
                max_val = ur((1+r)*a_grid[i]+b,γ,σ)
            else
                for i_next = 1:a_length # iterate over assets tomorrow choices
                    v_next = output.valfunc[i_next,j+1,1]
                    val = ur((1+r)*a_grid[i]+b-a_grid[i_next],γ,σ) + β*v_next
                    if val > max_val
                        max_val = val
                        output.polfunc[i,j,1] = a_grid[i_next]
                        output.polfunc[i,j,2] = a_grid[i_next]
                    end
                end
            end
            output.valfunc[i,j,1] = max_val
            output.valfunc[i,j,2] = max_val
        end
    end    
end

function worker(input::Input,output::Output;updating::Bool=false)
    @unpack γ, σ, a_length, a_grid, N, Jr, r, b, β, w, θ, eta, z, Π = input

    if updating == true
        r = output.r_SS
        w = output.w_SS
    end

    for j = (Jr-1):-1:1 # looping backwards from retirement to work
        for i_z = 1:2
            for i = 1:a_length # first loop over asset states
                max_val = -Inf # set our current max val to a low number
                for i_next = 1:a_length # loop over assets tomorrow
                    v_next = Π[i_z,1] * output.valfunc[i_next,j+1,1] + Π[i_z,2] * output.valfunc[i_next,j+1,2]
                    e = eta[j] * z[i_z]  # add age efficiency
                    l = ((γ*(1-θ)*e*w) - ((1-γ)*((1+r)*a_grid[i]-a_grid[i_next]))) / ((1-θ)*w*e) # labor
                    l = min(1.0,max(0.0,l))
                    val = uw(w*(1-θ)*e*l+(1+r)*a_grid[i]-a_grid[i_next],l,γ,σ) + β*v_next
                    if val > max_val
                        max_val = val
                        output.polfunc[i,j,i_z] = a_grid[i_next]
                        output.labfunc[i,j,i_z] = l
                    end
                end
                output.valfunc[i,j,i_z] = max_val
            end
        end
    end    
end

# Compute the steady-state distribution of agents over age (j), productivity (z), asset holdings (a): F_j(z, a)
function distribution(input::Input, output::Output)
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

    output.F = F
end

# Update current guess for steady-state aggregate labor and capital K_SS and L_SS
function K_L_update(input::Input, output::Output)
    @unpack θ, Jr, N, a_grid, a_length, z_length, eta, z = input
    #@unpack w_SS, L_SS, μ_age, F, labfunc = output

    # compute new guess for steady-state capital -- SHOULD VECTORIZE THIS
    K_next = sum(output.F .* a_grid)
    
    # yields same answer, but commenting out b/c not vectorized, so it's slow when we iterate
    # K_next_check = 0
    # for j=1:N, i=1:a_length, z=1:z_length
    #     K_next_check += F[i, j, z] * a_grid[i]
    # end

    # e[z, j]: earnings in state z at age j=1, ..., (Jr - 1) (2 x 45 matrix)
    e = eta' .* z

    # compute new guess for steady-state labor
    sum1 = sum(output.labfunc[:, :, 1] .* e[1, :]' .* output.F[:, 1:(Jr-1), 1]) # dot product for state z=1
    sum2 = sum(output.labfunc[:, :, 2] .* e[2, :]' .* output.F[:, 1:(Jr-1), 2]) # dot product for state z=2
    L_next = sum1 + sum2

    # yields same answer, but commenting out b/c not vectorized
    # L_next_check = 0
    # for j=1:(Jr-1), i=1:a_length, z=1:z_length
    #     L_next_check += F[i, j, z]*e[z, j]*labfunc[i, j, z]
    # end

    return K_next, L_next
end

# Repeatedly update guesses for K and L until convergence, then compute SS pension benefit + wages
function K_L_iterate(input::Input, output::Output; tol::Float64=1e-3, maxiter::Int64=100)
    @unpack α, Jr, N, θ = input
    
    err = 1000.0

    counter = 0

    while err > tol && counter < maxiter
        if mod(counter,5) == 0
            println("***** Iteration $counter:")
        end
        
        # (1) Solve retiree and worker problems with current prices
        #println("Solving retiree problem")
        retiree(input, output; updating=true)
        
        #println("Solving worker problem")
        worker(input, output; updating=true)
        
        # (2) Compute distribution of assets and productivity states across ages
        #println("Solving for stationary distribution F")
        distribution(input, output)
        
        # (3) Compute aggregate K and L implied by results from (1) and (2)
        #println("Updating capital and labor guesses")
        K_next, L_next = K_L_update(input, output)

        # Calculate difference b/w prev and current guesses for SS K and L
        err = max(abs(K_next - output.K_SS), abs(L_next - output.L_SS))
        if mod(counter,5) == 0
            println("Error at iteration $counter: " * string(err))
        end
        
        # Update guesses for SS K and L
        output.K_SS = 0.3 * K_next + 0.7 * output.K_SS
        output.L_SS = 0.3 * L_next + 0.7 * output.L_SS

        # Update prices based on updated K and L guesses
        output.w_SS = (1-α) * (output.K_SS^α) * (output.L_SS^(-α)) # wage = MPL
        output.r_SS = α * (output.K_SS^(α - 1)) * (output.L_SS^(1 - α)) # interest = MPK
        output.b_SS = (θ * output.w_SS * output.L_SS) / sum(output.μ_age[Jr:N])

        counter += 1
    end

    if counter == maxiter
        println("Reached max number of iterations ($maxiter) for K, L convergence.")
    else
        println("Guesses for K_SS and L_SS converged in $counter iterations!")
    end

    # Calculate welfare
    welfare = output.valfunc .* output.F
    welfare = sum(welfare[isfinite.(welfare)])
    output.welfare = welfare
end
