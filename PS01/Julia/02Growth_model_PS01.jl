#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    θ::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 75.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid

    #extending the model
    z_space::Array{Float64,1} = [1.25, 0.2] #Markov chain state space (2 states, good and bad)
    nz::Int64 = length(z_space) #number of states
    Π::Array{Float64,2} = [0.977 0.023; 0.074 0.926] #Markov chain transition matrix
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function: 2-dim b/c value fn differs by productivity state
    pol_func::Array{Float64, 2} #policy function: 2-dim b/c value fn differs by productivity state
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primitives (creates object of type "Primitives")
    val_func = zeros(prim.nk, prim.nz) #initial value function guess (nk x nz)
    pol_func = zeros(prim.nk, prim.nz) #initial policy function guess (nk x nz)
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack k_grid, β, δ, θ, nk, z_space, nz, Π = prim #unpack model primitives
    v_next = zeros(nk, nz) #next guess of value function to fill

    for z_index = 1:nz #iterating over productivity states (z)
        for k_index = 1:nk #iterating over values of k and solving for optimal k'(k)
            k = k_grid[k_index] #value of k
            candidate_max = -Inf #bad candidate max
            budget = z_space[z_index]*k^θ + (1-δ)*k #budget--add productivity shock

            for kp_index in 1:nk #loop over possible selections of k', exploiting monotonicity of policy function
                c = budget - k_grid[kp_index] #consumption given k' selection
                if c>0 #check for positivity
                    #compute expected value:
                    val = log(c) + β*(Π[z_index, 1]*val_func[kp_index, 1] + Π[z_index, 2]*val_func[kp_index, 2])
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[k_index, z_index] = k_grid[kp_index] #update policy function
                    end
                end
            end
            v_next[k_index, z_index] = candidate_max #update value function for given k,z
        end #end loop over values of k
    end #end loop over Markov chain states
    v_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        #err = abs.(maximum(v_next.-res.val_func))/abs(v_next[prim.nk, 1]) #reset error level
        err = maximum(abs.(v_next.-res.val_func)) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
