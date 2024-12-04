@concrete mutable struct NonlinearSolveForwardDiffCache <: AbstractNonlinearSolveCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end
