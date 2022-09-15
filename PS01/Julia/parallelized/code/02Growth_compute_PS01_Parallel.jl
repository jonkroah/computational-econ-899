using Distributed # parallel processing package
addprocs(2)

@everywhere using Parameters, Plots, SharedArrays #import the libraries we want

cd("C:/Users/jgkro/Documents/GitHub/computational-econ-899/PS01/Julia/parallelized")

@everywhere include("02Growth_model_PS01_Parallel.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value functions
Plots.plot(k_grid, val_func, title="Value Function", label = ["Good State" "Bad State"])
Plots.savefig("output/02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions", label = ["Good State" "Bad State"])
Plots.savefig("output/02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes", label = ["Good State" "Bad State"])
Plots.savefig("output/02_Policy_Functions_Changes.png")

println("All done!")
################################
