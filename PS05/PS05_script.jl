cd("C:\\Users\\jgkro\\Documents\\GitHub\\compdjj\\PS5\\jon")

include("PS05_model.jl")

# Initialize
prim = Primitives()
shocks = Shocks()
output = initialize_output(prim)

# Simulate panel of (ε, z) shocks
ε_state, z_state = draw_shocks(shocks, prim.N, prim.T)

# Solve the model
solve_model(prim, shocks, output, ε_state, z_state)