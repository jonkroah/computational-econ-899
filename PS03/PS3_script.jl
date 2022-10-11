# Computational Economics - Problem Set 3
# Authors: Dalya, Jackson, Jon
# September 2022

# This is the main file - run this one.

println("Starting...")

# You may need to manually set your file path in the Julia terminal using pwd("C:\\Example\\Filepath")
# Or you can change this line of code:
cd("C:\\Users\\jgkro\\Documents\\GitHub\\computational-econ-899\\PS03")
#cd("/Users/dalya/Documents/GitHub/compdjj/PS3")
# cd("C:\\Users\\jaxtc\\OneDrive\\Documents\\GitHub\\compdjj\\PS3")

# Bring in model and other functions
include("PS3_model.jl")
#include("test.jl")

# Initialize input (primitives) and output (solutions) structures
input = Input()
output = Initialize(input)

retiree(input,output)
worker(input,output)
println("Solved retiree and worker problems.")
output.valfunc[:, 66, 1]

# value functions
using Plots
@unpack a_grid = input
plot(a_grid, [output.valfunc[:,20,1],output.valfunc[:,20,2],
    output.valfunc[:,50,1]], 
    labels=["N=20 (high z)" "N=20 (low z)" "N=50"], title="Value Functions", 
    xlabel="a", ylabel="V(a,s)", legend=:topleft, linewidth=2)
savefig("PS03_valfunc_2050.png")

# policy functions
plot(a_grid, [output.polfunc[:,20,1],output.polfunc[:,20,2],
    output.polfunc[:,50,1]], 
    labels=["N=20 (high z)" "N=20 (low z)" "N=50"], 
    title="Savings Functions", xlabel="a", ylabel="g(a,s)", 
    legend=:topleft, linewidth=2)
savefig("PS03_polfunc_2050.png")

# compute stationary distribution F_j(a,z)
distribution(input,output)

#=

# iterate to get SS capital, labor, prices, pension benefit
K_L_iterate(input, output)

io = open("benchSS.txt","w")

# results - benchmark
println(io,"Benchmark model w/ SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)
io = open("benchNOSS.txt","w")

# Repeat, benchmark w/o social security
input = Input(θ=0)
output = Initialize(input)
retiree(input,output)
worker(input,output)
#println("Solved retiree and worker problems.")
distribution(input,output)
K_L_iterate(input, output)
# results - benchmark w/o social security
println(io,"Benchmark model w/o SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)

=#

io = open("noriskSS.txt","w")

# Repeat, no risk w/ social security
input = Input(z=[0.5, 0.5])
output = Initialize(input)
retiree(input,output)
worker(input,output)
#println("Solved retiree and worker problems.")
distribution(input,output)
K_L_iterate(input, output)
# results - no risk w/ social security
println(io,"No risk model w/ SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)
io = open("noriskNOSS.txt","w")

# Repeat, no risk w/o social security
input = Input(z=[0.5, 0.5],θ=0)
output = Initialize(input)
retiree(input,output)
worker(input,output)
#println("Solved retiree and worker problems.")
distribution(input,output)
K_L_iterate(input, output)
# results - no risk w/o social security
println(io,"No risk model w/o SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)
io = open("exoglsSS.txt","w")

# Repeat, exog LS w/ social security
input = Input(γ=1)
output = Initialize(input)
retiree(input,output)
worker(input,output)
#println("Solved retiree and worker problems.")
distribution(input,output)
K_L_iterate(input, output)
# results - exog LS w/ social security
println(io,"Exog. LS model w/ SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)
io = open("exoglsNOSS.txt","w")

# Repeat, exog LS w/o social security
input = Input(γ=1,θ=0)
output = Initialize(input)
retiree(input,output)
worker(input,output)
#println("Solved retiree and worker problems.")
distribution(input,output)
K_L_iterate(input, output)
# results - exog LS w/o social security
println(io,"Exog. LS model w/o SS:")
println(io,"Reminder, population growth = $(input.n)")
println(io,"capital K: $(output.K_SS)")
println(io,"labor L: $(output.L_SS)")
println(io,"wage w: $(output.w_SS)")
println(io,"interest r: $(output.r_SS)")
println(io,"pension benefit b: $(output.b_SS)")
println(io,"welfare: $(output.welfare)")

close(io)

println("Done!")