using Parameters, XLSX, DataFrames, Optim, LaTeXTabulars

cd("/Users/jonkroah/Documents/GitHub/computational-econ-899/PS08")

#-----------------------------------------------------------------
# (0) Read in data, build matrices for MLE, store in struct
#-----------------------------------------------------------------
@with_kw struct Data
    mortgage_data::DataFrame
    y::Vector{Int64}
    x::Matrix{Float64}
    n::Int64
    k::Int64
    x_labels::Array{String}
end

function prep_data()
    # Read in data
    mortgage_data = DataFrame(XLSX.readtable("data/Mortgage_performance_data.xlsx", "Sheet1"))

    # Choices (0 or 1)
    y = convert(Vector{Int64}, mortgage_data.i_close_first_year)

    # Covariates
    x = convert(Matrix{Float64},
            Matrix(select(mortgage_data,
                    [:i_large_loan   ,
                    :i_medium_loan 	,
                    :rate_spread 	,
                    :i_refinance 	,
                    :age_r 			,
                    :cltv 			,
                    :dti 			,
                    :cu 			,
                    :first_mort_r 	,
                    :score_0 		,
                    :score_1 		,
                    :i_FHA 			,
                    :i_open_year2 	,
                    :i_open_year3 	,
                    :i_open_year4 	,
                    :i_open_year5]
                    )))

    # x labels (including intercept)
    x_labels = ["Intercept", "Large loan", "Medium loan", "Rate spread", "Refi", "Age" , "Combined LTV" , "Mtg. Debt-to-Income", "Credit union", "First mtg.", "FICO year 0", "FICO year 1", "FHA", "Open 2014", "Open 2015", "Open 2016", "Open 2017"]
    
    n = nrow(mortgage_data)  # number of observations
    k = size(x)[2]           # number of covariates excluding intercept

    data = Data(mortgage_data, y, x, n, k, x_labels)
    return data
end

#-----------------------------------------------------------------
# (1.1) Log-likelihood
#   Inputs:
#       Parameters β = (β_0, β_1)
#       Vector of choices y_1, ..., y_n
#       Array of covariates x_1, ..., x_n
#   Output:
#       Value of the log-likelihood function (scalar)
#-----------------------------------------------------------------
function log_likelihood(β::Vector{Float64}, y::Vector{Int64}, x::Array{Float64})
    n = length(y)
    @assert n == length(x[:, 1]) "Error: Y and X have different numbers of rows."

    # Extract intercept & coefficients on covariates
    β_0 = β[1]
    β_1 = β[2:end]

    # Pr(Y = 1 | X = x, β)
    p_1 = exp.(β_0 .+ x * β_1) ./ (1 .+ exp.(β_0 .+ x * β_1))

    # Pr(Y = 0 | X = x, β)
    p_0 = 1 .- p_1

    # Pr(Y = y | X = x, β)
    p = p_1 .^ (y .== 1) .* p_0 .^ (y .== 0)

    # Log-likelihood: sum of log(Pr(Y = y | X = x, β))
    L = sum(log.(p))

    return L
end

#--------------------------------------------------
# (1.2) Score (gradient of the log-likelihood)
#   Inputs:
#       Parameters β = (β_0, β_1)
#       Vector of choices y_1, ..., y_n
#       Array of covariates x_1, ..., x_n
#   Output:
#       Value of the score: (K+1)x1 vector
#--------------------------------------------------
function score(β::Vector{Float64}, y::Vector{Int64}, x::Array{Float64})
    # Extract intercept & coefficients on covariates
    β_0 = β[1]
    β_1 = β[2:end]
    
    # choice probabilities
    p_1 = exp.(β_0 .+ x * β_1) ./ (1 .+ exp.(β_0 .+ x * β_1))
    p_0 = 1 .- p_1
    
    # Partial derivative w.r.t. β_0
    d0 = sum((y .== 1) .- p_1)

    # Partial derivative w.r.t. β_1
    d1 = x' * ((y .== 1) .- p_1)
    
    s = vcat(d0, d1)

    @assert size(s) == (length(β),)

    return s
end

#--------------------------------------------------
# (1.3) Hessian
#   Inputs:
#       Parameters β = (β_0, β_1)
#       Vector of choices y_1, ..., y_n
#       Array of covariates x_1, ..., x_n
#   Output:
#       Value of the Hessian (matrix)
#--------------------------------------------------
function hessian(β::Vector{Float64}, y::Vector{Int64}, x::Array{Float64})
    # Extract intercept & coefficients on covariates
    β_0 = β[1]
    β_1 = β[2:end]

    # choice probabilities
    p_1 = exp.(β_0 .+ x * β_1) ./ (1 .+ exp.(β_0 .+ x * β_1))
    p_0 = 1 .- p_1
    
    # ∂²l / ∂β_0²
    d00 = -sum(p_1 .* p_0)
    
    # ∂²l / ∂β_1 ∂β_1'
    d11 = -x' * (x .* p_1 .* p_0)

    # ∂²l / ∂β_0 ∂β_1'
    d01 = -x' * (p_1 .* p_0)

    # build Hessian & return
    H = [d00 d01';
        d01 d11]

    # Make sure dimensions are correct
    @assert size(H) == (length(β), length(β))

    return H
end

#--------------------------------------------------
# (2) Numerical derivatives
#--------------------------------------------------
#--------------------------------------------------
# (2.1) Numerical gradient
#--------------------------------------------------
function numerical_gradient(f::Function, β::Vector{Float64}; epsilon::Float64 = 1e-5)
    k = length(β)
    gradient = zeros(k)

    for j=1:k
        β_pert_j = vcat(β[1:j-1], β[j] + epsilon, β[j+1:end])
        β_pert_j_neg = vcat(β[1:j-1], β[j] - epsilon, β[j+1:end])
        # gradient[j] = (f(β_pert_j) - f(β)) / epsilon
        gradient[j] = (f(β_pert_j) - f(β_pert_j_neg)) / (2*epsilon)
    end
    
    return gradient
end

#--------------------------------------------------
# (2.2) Numerical Hessian
#--------------------------------------------------
function numerical_hessian(f::Function, β::Vector{Float64}; epsilon::Float64 = 1e-5)
    k = length(β)

    hessian = zeros(k,k)

    for j=1:k, l=1:k
        β_pert_j    = vcat(β[1:j-1], β[j] + epsilon, β[j+1:end])
        β_pert_l    = vcat(β[1:l-1], β[l] + epsilon, β[l+1:end])
        β_pert_jl   = vcat(β_pert_j[1:l-1], β_pert_j[l] + epsilon, β_pert_j[l+1:end])

        temp1 = (f(β_pert_jl) - f(β_pert_l)) / epsilon
        temp2 = (f(β_pert_j) - f(β)) / epsilon
        
        hessian[j,l] = (temp1 - temp2) / epsilon
    end

    return hessian
end

#--------------------------------------------------
# (3) Newton's method
#--------------------------------------------------
function newton_solve(grad::Function, hess::Function, guess::Vector{Float64}; tol::Float64=10e-12, maxiter::Int64=100)
    # Initialize guess & other stuff
    β_current = guess
    err = 1000.0
    counter = 0

    while err >= tol && counter < maxiter
        β_next = β_current - inv(hess(β_current)) * grad(β_current)

        err = maximum(abs.(β_next .- β_current))
        println("Diff b/w guess $(counter) and $(counter+1): $err")

        β_current = β_next
        counter +=1
    end
    if counter == maxiter
        println("Reached max number of iterations ($maxiter). Error=$err, tolerance=$tol.")
    elseif err < tol
        println("Newton algorithm converged in $counter steps! Error=$err, tolerance=$tol.")
    end

    return β_current
end

