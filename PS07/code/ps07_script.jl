cd("/Users/jonkroah/Documents/GitHub/computational-econ-899/PS07")
include("ps07_functions.jl")
Random.seed!(1234)

# Initialize data & draws
prim, truedata, simdata = initialize()
@unpack x = truedata

# Weighting matrices
I2 = [1.0 0.0; 0.0 1.0]
I3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

# Grids for graphing
ρ_grid = range(start=0.35, stop=0.65, length=100)
σ_grid = range(start=0.8, stop=1.2, length=100)

#--------------------------------------------------------------------
# Question 4: use mean & variance (moments 1 & 2)
#--------------------------------------------------------------------
moments_to_use = [1, 2]

# Question 4(a): Plot GMM objective function over (ρ,σ)
# Use first two moments to form objective function
gmm_obj_grid = zeros(length(ρ_grid), length(σ_grid))
for (i_ρ, ρ) in enumerate(ρ_grid)
    for (i_σ, σ) in enumerate(σ_grid)
        gmm_obj_grid[i_ρ, i_σ] = gmm_obj([ρ, σ], moments_to_use, I2, prim, simdata, truedata)
    end
end
Plots.contourf(ρ_grid, σ_grid, gmm_obj_grid, xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_1_2_contour")

Plots.surface(ρ_grid, σ_grid, gmm_obj_grid, camera=(35,15), xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_1_2_surface")

# 1-step and 2-step GMM
b_hat_step1, b_hat_step2, S_TH, W_step2, gradient, vcv, se_hat, J_stat = gmm_2step(moments_to_use, [0.4, 0.8], prim, simdata, truedata)

open("output/gmm_results_moments_1_2.txt", "w") do file
    write(file, "b_hat_step1 = $(round.(b_hat_step1, digits=4))")
    write(file, "\n")
    write(file, "b_hat_step2 = $(round.(b_hat_step2, digits=4))")
    write(file, "\n")
    write(file, "S_TH = $(round.(S_TH, digits=4))")
    write(file, "\n")
    write(file, "W_step2 = $(round.(W_step2, digits=4))")
    write(file, "\n")
    write(file, "Gradient (dg_dρ, dg_dσ) = $(round.(gradient, digits=4))")
    write(file, "\n")
    write(file, "VCV for b_hat_step2 = $(round.(vcv, digits=4))")
    write(file, "\n")
    write(file, "SE(b_hat_step2) = $(round.(se_hat, digits=4))")
    write(file, "\n")
    write(file, "J_stat = $(round.(J_stat, digits=9))")
end

#--------------------------------------------------------------------
# Question 5: use variance & autocorr (moments 2 & 3)
#--------------------------------------------------------------------
moments_to_use = [2, 3]

# Question 5(a): Plot GMM objective function over (ρ,σ)
# Use first two moments to form objective function
gmm_obj_grid = zeros(length(ρ_grid), length(σ_grid))
for (i_ρ, ρ) in enumerate(ρ_grid)
    for (i_σ, σ) in enumerate(σ_grid)
        gmm_obj_grid[i_ρ, i_σ] = gmm_obj([ρ, σ], moments_to_use, I2, prim, simdata, truedata)
    end
end
Plots.contourf(ρ_grid, σ_grid, gmm_obj_grid, xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_2_3_contour")

Plots.surface(ρ_grid, σ_grid, gmm_obj_grid, camera=(35,15), xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_2_3_surface")

# 1-step and 2-step GMM
b_hat_step1, b_hat_step2, S_TH, W_step2, gradient, vcv, se_hat, J_stat = gmm_2step(moments_to_use, [0.4, 0.8], prim, simdata, truedata)

open("output/gmm_results_moments_2_3.txt", "w") do file
    write(file, "b_hat_step1 = $(round.(b_hat_step1, digits=4))")
    write(file, "\n")
    write(file, "b_hat_step2 = $(round.(b_hat_step2, digits=4))")
    write(file, "\n")
    write(file, "S_TH = $(round.(S_TH, digits=4))")
    write(file, "\n")
    write(file, "W_step2 = $(round.(W_step2, digits=4))")
    write(file, "\n")
    write(file, "Gradient (dg_dρ, dg_dσ) = $(round.(gradient, digits=4))")
    write(file, "\n")
    write(file, "VCV for b_hat_step2 = $(round.(vcv, digits=4))")
    write(file, "\n")
    write(file, "SE(b_hat_step2) = $(round.(se_hat, digits=4))")
    write(file, "\n")
    write(file, "J_stat = $(round.(J_stat, digits=9))")
end

#--------------------------------------------------------------------
# Question 5: use mean, variance, & autocorr (moments 1, 2, 3)
#--------------------------------------------------------------------
moments_to_use = [1, 2, 3]

# Question 5(a): Plot GMM objective function over (ρ,σ)
# Use first two moments to form objective function
gmm_obj_grid = zeros(length(ρ_grid), length(σ_grid))
for (i_ρ, ρ) in enumerate(ρ_grid)
    for (i_σ, σ) in enumerate(σ_grid)
        gmm_obj_grid[i_ρ, i_σ] = gmm_obj([ρ, σ], moments_to_use, I3, prim, simdata, truedata)
    end
end
Plots.contourf(ρ_grid, σ_grid, gmm_obj_grid, xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_1_2_3_contour")

Plots.surface(ρ_grid, σ_grid, gmm_obj_grid, camera=(35,15), xlabel="ρ", ylabel="σ", seriescolor=:viridis)
png("output/gmm_moments_1_2_3_surface")

# 1-step and 2-step GMM
b_hat_step1, b_hat_step2, S_TH, W_step2, gradient, vcv, se_hat, J_stat = gmm_2step(moments_to_use, [0.4, 0.8], prim, simdata, truedata)

open("output/gmm_results_moments_1_2_3.txt", "w") do file
    write(file, "b_hat_step1 = $(round.(b_hat_step1, digits=4))")
    write(file, "\n")
    write(file, "b_hat_step2 = $(round.(b_hat_step2, digits=4))")
    write(file, "\n")
    write(file, "S_TH = $(round.(S_TH, digits=4))")
    write(file, "\n")
    write(file, "W_step2 = $(round.(W_step2, digits=4))")
    write(file, "\n")
    write(file, "Gradient (dg_dρ, dg_dσ) = $(round.(gradient, digits=4))")
    write(file, "\n")
    write(file, "VCV for b_hat_step2 = $(round.(vcv, digits=4))")
    write(file, "\n")
    write(file, "SE(b_hat_step2) = $(round.(se_hat, digits=4))")
    write(file, "\n")
    write(file, "J_stat = $(round.(J_stat, digits=9))")
end