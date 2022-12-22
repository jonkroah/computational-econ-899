using Plots

q11 = 0.5
q21 = -1.0
q12 = -1.0
q22 = 0.5

q1 = 1.0
q2 = 1.0

derivs = [q11 q21; q12 q22]

s_wic_grid = 0.0:0.01:1.0
margins = zeros(2, length(s_wic_grid))

for (i, s_wic) in enumerate(s_wic_grid)
    s = [s1 + s_wic, s2]
    margins[:, i] = -inv(derivs) * s
end

plot(s_wic_grid, margins', legend=:topleft, labels=["Subsidized" "Unsubsidized"])