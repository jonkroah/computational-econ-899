cd("/Users/jonkroah/Documents/GitHub/computational-econ-899/JF_PS05")
include("ps05_functions.jl")

# Solve model
prim, results = solve_model()

# Simulate industry path, using model estimates
industry_path_sim((0.0, 0.0), 25, 1000)

@unpack q_grid, q_grid_length, ω, ω_length = prim
@unpack q_star_1, q_star_2, π_star_1, π_star_2, valfunc_1, valfunc_2, x_polfunc_1, x_polfunc_2, ω_paths_sim = results

# Cournot quantities
Plots.surface(
    q_grid, q_grid, q_star_1',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="q_star_1",
    seriescolor=:viridis, camera=(30,20), legend=:false
)
png("output/cournot_q_star_1")

Plots.surface(
    q_grid, q_grid, q_star_2',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="q_star_2",
    seriescolor=:viridis, camera=(30,20), legend=:false
)
png("output/cournot_q_star_2")


# Static Cournot payoffs
Plots.surface(
    q_grid, q_grid, π_star_1',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="π_star_1",
    seriescolor=:viridis, camera=(30,20), legend=:false
)
png("output/cournot_payoffs_1")

Plots.surface(
    q_grid, q_grid, π_star_2',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="π_star_2",
    seriescolor=:viridis, camera=(30,20), legend=:false
    )
png("output/cournot_payoffs_2")


# Policy functions
Plots.surface(
    q_grid, q_grid, x_polfunc_1',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="x_1",
    seriescolor=:viridis, camera=(30,20), legend=:false
)
png("output/x_polfunc_1")

Plots.surface(
    q_grid, q_grid, x_polfunc_2',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="x_2",
    seriescolor=:viridis, camera=(30,20), legend=:false
    )    
png("output/x_polfunc_2")

# Value functions
Plots.surface(
    q_grid, q_grid, valfunc_1',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="V_1",
    seriescolor=:viridis, camera=(30,20), legend=:false
)
png("output/valfunc_1")

Plots.surface(
    q_grid, q_grid, valfunc_2',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="V_2",
    seriescolor=:viridis, camera=(30,20), legend=:false
    )
png("output/valfunc_2")


# Joint distribution of ω(t=25) in simulations
ω_25 = ω[ω_paths_sim][25, :]

q_bar_25_density = zeros(q_grid_length, q_grid_length)
for rep=1:length(ω_25)
    i1 = findfirst(x->x == ω_25[rep][1], q_grid)
    i2 = findfirst(x->x == ω_25[rep][2], q_grid)

    q_bar_25_density[i1, i2] += 1.0
end
q_bar_25_density ./=  length(ω_25)

Plots.surface(
    q_grid, q_grid, q_bar_25_density',
    xlabel="q_bar_1", ylabel="q_bar_2", zlabel="Density",
    seriescolor=:viridis, camera=(30,30), legend=:false
)
png("output/t25_state_hist")


# Make table w/ Cournot equilibria
q_bar = zeros(Int(((length(q_grid) * (length(q_grid) + 1)) / 2)), 2)
q_star = zeros(Int(((length(q_grid) * (length(q_grid) + 1)) / 2)), 2)
payoffs = zeros(Int(((length(q_grid) * (length(q_grid) + 1)) / 2)), 2)

count = 1
for i1=1:q_grid_length, i2=1:q_grid_length
    if i1 >= i2
        q_bar[count, 1] = q_grid[i1]
        q_bar[count, 2] = q_grid[i2]
        
        q_star[count, 1] = q_star_1[i1, i2]
        q_star[count, 2] = q_star_2[i1, i2]

        payoffs[count, 1] = π_star_1[i1, i2]
        payoffs[count, 2] = π_star_2[i1, i2]
        count += 1
    end
end

lap(hcat(q_bar, round.(q_star, digits=3), round.(payoffs, digits=3))) # copy from terminal to clipboard