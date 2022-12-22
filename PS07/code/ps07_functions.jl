using Random, Distributions, Parameters, Plots, LinearAlgebra, Optim

@with_kw struct Primitives
    ρ0::Float64
    σ0::Float64
    x0::Float64
    T::Int64
    H::Int64
end

@with_kw struct TrueData
    x::Array{Float64, 1}
end

mutable struct SimData
    e_std::Array{Float64, 2}
end

#---------Construct true data
function construct_true_data(prim::Primitives)
    @unpack ρ0, σ0, x0, T = prim

    ε_dist = Normal(0, σ0^1)
    ε = rand(ε_dist, T + 1)

    x = zeros(T + 1)
    for t=1:(T + 1)
        if t == 1
            x[t] = x0
        else
            x[t] = ρ0 * x[t - 1] + ε[t]
        end
    end

    return x
end

#---------Initialize things:
#---Model primitives and number of simulation draws
#---Sequence of "true" data
#---Random draws to be used for all simulations
function initialize()
    ρ0      = 0.5
    σ0      = 1.0
    x0      = 0.0
    T       = 200
    H       = 10
    prim    = Primitives(ρ0, σ0, x0, T, H)

    # Construct true data
    truedata = TrueData(construct_true_data(prim))

    # Draw shocks to be used in all simulations
    e_std_dist  = Normal(0, 1)
    e_std       = rand(e_std_dist, T + 1, H)
    y           = zeros(T + 1, H)
    simdata     = SimData(e_std)

    return prim, truedata, simdata
end

#---------Construct simulated data, given parameters
function construct_sim_data(ρ::Float64, σ::Float64, prim::Primitives, simdata::SimData)
    @unpack e_std = simdata
    @unpack T, H = prim

    e = σ .* e_std

    y = zeros(T + 1, H)
    for t=1:(T + 1), h=1:H
        if t == 1
            y[t, h] = 0.0
        else
            y[t, h] = ρ * y[t - 1, h] + e[t, h]
        end
    end

    return y
end


#----------------------------------------------
#--- Compute m = [z_t, (z_t - z_bar)^2, (z_t - z_bar)(z_t-1 - z_bar)]
#--- Only return the elements corresponding to the moments
#       we'll use in estimation
function m(z::Array{Float64}, moments_to_use::Vector{Int64})
    m1 = z

    m2 = (z .- mean(z, dims=1)).^2

    z_lag = vcat(0.0, z[1:(end-1), :])

    if typeof(z) == Vector{Float64}
        m3 = collect(skipmissing(
            (z .- mean(z, dims=1)) .* (z_lag .- mean(z, dims=1))
            ))
    elseif typeof(z) == Matrix{Float64}
        z_dims = size(z)
        m3 = reshape(
            collect(skipmissing(
            (z .- mean(z, dims=1)) .* (z_lag .- mean(z, dims=1))
            ))
            , z_dims[1], z_dims[2])
        # need to wrap skipmissing() in collect(), but collect() 
        #   transforms the data into a vector
        # then need to reshape this vector according to the dimensions
        #   of the original matrix z (minus 1 row to account for the lag)
    end

    temp = [m1, m2, m3]

    m = []
    for i_m in moments_to_use
        push!(m, temp[i_m])
    end

    return m
end

#-------Function to compute moments for specific observations
# Input: y_th
# Output: [y_th, ]

#-------Function to compute M_T(x) and M_TH(y)
# Argument = vector m(z)
# Output = vector of moments M(x)
function moments(m_z::Vector{Any})
    moments = zeros(length(m_z))
    for i=1:length(m_z)
        moments[i] = mean(m_z[i])
    end
    return moments
end




#-------- Function to compute Γ_TH(j)
function Γ_TH(j::Int64, y::Array{Float64, 2}, moments_to_use::Vector{Int64})
    T       = size(y)[1]
    H       = size(y)[2]
    m_y     = m(y, moments_to_use)
    M_TH    = moments(m(y, moments_to_use))

    Γ = zeros(length(moments_to_use), length(moments_to_use))
    for h = 1:H, t=j:T
        temp = zeros(length(moments_to_use))
        for (i_m, moment) in enumerate(moments_to_use)
            # println("h=$h, t=$t, i_m=$i_m, moment=$moment:")
            temp[i_m] = m_y[i_m][t, h] - M_TH[i_m]
        end
        Γ += temp * temp'
    end

    Γ = Γ ./ (T * H)

    return Γ
end

#-----Function to compute S_y_TH
function S_matrices(y::Array{Float64, 2}, i_T::Int64, moments_to_use::Vector{Int64})
    S_y_TH  = Γ_TH(1, y, moments_to_use)
    
    for j=2:(i_T + 1)
        S_y_TH += (1 - j / (i_T + 1)) * (Γ_TH(j, y, moments_to_use) + Γ_TH(j, y, moments_to_use)')
    end

    H = size(y)[2]
    S_TH = (1 + (1 / H)) * S_y_TH

    return S_y_TH, S_TH
end


#------GMM objective function
# Argument 'moments' is a vector of integers corresponding to the moments we want to use
#   (1=mean, 2=variance, 3=first-order autocorrelation)
function gmm_obj(b::Vector{Float64}, moments_to_use::Vector{Int64}, W::Matrix{Float64}, prim::Primitives, simdata::SimData, truedata::TrueData)
    ρ, σ = b[1], b[2]
    
    y = construct_sim_data(ρ, σ, prim, simdata)

    @unpack x = truedata

    # Compute moments and create vector of relevant moments
    M_T = moments( m(x, moments_to_use) )
    M_TH = moments( m(y, moments_to_use) )

    # Form criterion function
    g       = M_T .- M_TH
    J_TH    = g' * W * g

    # Return
    return J_TH
end

#-------Function to do 2-step GMM (outputs 1-step results)
function gmm_2step(moments_to_use::Vector{Int64}, guess::Vector{Float64},
    prim::Primitives, simdata::SimData, truedata::TrueData)
    
    @unpack T, H = prim

    W_step1 = Matrix(Diagonal(ones(length(moments_to_use))))

    gmm_step1 = optimize(
        b -> gmm_obj(b, moments_to_use, W_step1, prim, simdata, truedata),
        guess,
        method = NelderMead(),
        iterations = 10_000
    )

    b_hat_step1 = gmm_step1.minimizer

    # Estimate VCV & update weighting matrix
    y_update = construct_sim_data(b_hat_step1[1], b_hat_step1[2], prim, simdata)
    S_y_TH, S_TH = S_matrices(y_update, 4, moments_to_use)
    W_step2 = inv(S_TH)

    # Run GMM again
    gmm_step2 = optimize(
        b -> gmm_obj(b, moments_to_use, W_step2, prim, simdata, truedata),
        b_hat_step1,
        method = NelderMead(),
        iterations = 10_000
    )
    b_hat_step2 = gmm_step2.minimizer
    J_TH = gmm_step2.minimum


    # Estimate gradient + VCV for b_hat_step2
    perturb = 0.00001
    y_pert_ρ = construct_sim_data(b_hat_step1[1] - perturb, b_hat_step1[2], prim, simdata)
    y_pert_σ = construct_sim_data(b_hat_step1[1], b_hat_step1[2] - perturb, prim, simdata)

    dg_dρ = -(moments(m(y_update, moments_to_use)) - moments(m(y_pert_ρ, moments_to_use))) ./ perturb

    dg_dσ = -(moments(m(y_update, moments_to_use)) - moments(m(y_pert_σ, moments_to_use))) ./ perturb

    gradient = hcat(dg_dρ, dg_dσ)
    
    vcv = inv(gradient' * W_step2 * gradient) ./ T

    # Compute se_hats
    se_hat = sqrt.(diag(vcv))

    # Compute J test statistic
    J_stat = (T * H / (1 + H)) * J_TH

    # Return estimates
    return b_hat_step1, b_hat_step2, S_TH, W_step2, gradient, vcv, se_hat, J_stat
end
