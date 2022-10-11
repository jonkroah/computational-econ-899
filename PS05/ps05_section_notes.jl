#=
Econ 899 Problem Set 5 -- section notes

--------True problem:
V(k, ε, μ, z) = max_{k'} {u(c) + β*E[V(k', ε', μ', z')]}
    s.t. BC: depends on k, ε, K, z

where μ is the distribution of HHs across (k, ε).

PROBLEM: Transition from μ to μ' is extremely complicated, depends on aggregate
uncertainty z --> z' and idiosyncratic uncertainty ε --> ε'.

--------Krusell-Smith:
Instead of μ,μ', think of K,K' (aggregates).
Approximate the true HH prob above as

V(k, ε, K, z) = max_{k'} {u(c) + β*E[V(k', ε', K', z')]}
    s.t. BC: depends on k, ε, K, z

Guess a transition function: K'(K, z)


=#