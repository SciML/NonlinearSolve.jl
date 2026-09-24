using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using SCCNonlinearSolve

# Two homogeneous nonlinear blocks in a `Vector{Any}` container: the solve
# must preserve the container eltype through `scc_solve_up`, so `_scc_solve`
# sees the same `SCCNonlinearProblem` type as for heterogeneous block vectors.
function anyvec_scc_residual!(resid, u, p)
    resid[1] = u[1]^2 - p[1]
    return nothing
end

_hp = [2.0]
_hq1 = NonlinearProblem(anyvec_scc_residual!, [1.0], _hp)
_hq2 = NonlinearProblem(anyvec_scc_residual!, [1.0], _hp)
_probs_any = Any[_hq1, _hq2]
@test _probs_any isa Vector{Any}
_sccprob = SciMLBase.SCCNonlinearProblem(
    _probs_any, Any[Returns(nothing), Returns(nothing)], _hp, true
)
@test _sccprob.probs isa Vector{Any}

_scc_alg = SCCNonlinearSolve.SCCAlg(; nlalg = NewtonRaphson(), linalg = nothing)
_sol = solve(_sccprob, _scc_alg)
@test SciMLBase.successful_retcode(_sol)
@test _sol.u ≈ [sqrt(2.0), sqrt(2.0)]

# The solution carries the concrete problem `_scc_solve` received: its block
# container must still be `Vector{Any}`, not narrowed to the homogeneous
# block type (which would be a different compiled solve path per homogeneity).
@test _sol.prob.probs isa Vector{Any}
