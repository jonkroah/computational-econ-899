cd("/Users/jonkroah/Documents/GitHub/computational-econ-899/PS08")

include("ps01_functions.jl")

#-------------------------------------------------------
# Load data
#-------------------------------------------------------
data = prep_data()
@unpack y, x, k, n, x_labels = data

#-------------------------------------------------------
# Test point to use throughout the pset
#-------------------------------------------------------
β_test = vcat([-1.0], zeros(k))

#-------------------------------------------------------
# Q1: evaluate log-likelihood, score, Hessian at
#   β_0 = -1, β_1 = (0, ..., 0)
#-------------------------------------------------------
# Row/column labels
x_num_labels = repeat(["("], 17) .* string.(0:16) .* repeat([")"], 17)
x_num_full_labels = hcat(x_num_labels, x_labels)
col_labels = hcat([""], permutedims(x_num_labels))

# Compute likelihood, score, Hessian
l = round.(log_likelihood(β_test, y, x), digits=3)
s = round.(score(β_test, y, x), digits=1)
h = round.(hessian(β_test, y, x), digits=1)

s = hcat(x_num_full_labels, s)

h = vcat(col_labels, hcat(x_num_labels, h))

# Print to .txt file
io = open("output/ps01_q01.txt","w")
println(io, "Log-likelihood: $l")
close(io)

# Print score and Hessian to tex files
latex_tabular("output/ps01_q01_score.tex", Tabular("llc"), s)
latex_tabular("output/ps01_q01_hessian.tex", Tabular("c"^(k+2)), h)

#-------------------------------------------------------
# Q2: numerical first and second derivatives
#-------------------------------------------------------
ns = round.(numerical_gradient(β -> log_likelihood(β, y, x), β_test), digits=1)
ns = hcat(x_num_labels, ns)

nh = round.(numerical_hessian(β -> log_likelihood(β, y, x), β_test), digits=1)
nh = vcat(col_labels, hcat(x_num_labels, nh))

# Print score and Hessian to tex files
# latex_tabular("output/ps01_q02_numerical_score.tex", Tabular("lc"), ns)
latex_tabular("output/ps01_q02_numerical_score.tex", Tabular("c"^(k+1)), permutedims(ns))
latex_tabular("output/ps01_q02_numerical_hessian.tex", Tabular("c"^(k+2)), nh)

#-------------------------------------------------------
# Q3 and Q4: MLE using Newton, BFGS, and Nelder-Mead
#-------------------------------------------------------
# Newton method using analytic derivatives
@time β_newton = newton_solve(β -> score(β, y, x), β -> hessian(β, y, x), β_test)

# Newton method using numerical derivatives -- this will cycle unless we reduce the tolerance a bit
# @time newton_solve(
#     β -> numerical_gradient(β -> log_likelihood(β, y, x), β; epsilon=1e-5),
#     β -> numerical_hessian(β -> log_likelihood(β, y, x), β; epsilon=1e-5),
#     β_test;
#     tol = 10e-10
#     )

# BFGS
@time res_BFGS = optimize(β -> -log_likelihood(β, y, x), β_test, method = BFGS(), iterations = 10_000)
β_BFGS = res_BFGS.minimizer

# Simplex
@time res_simplex = optimize(β -> -log_likelihood(β, y, x), β_test, method = NelderMead(), iterations = 10_000)
β_simplex = res_simplex.minimizer

# Print to tex table
latex_tabular("output/ps01_q03_q04.tex", Tabular("cccc"),
    vcat(
        ["k" "Newton" "BFGS" "Simplex"], 
        hcat(x_num_labels, round.(β_newton, digits = 3), round.(β_BFGS, digits = 3), round.(β_simplex, digits = 3))
    ))