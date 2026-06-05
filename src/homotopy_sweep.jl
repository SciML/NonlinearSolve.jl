"""
    HomotopySweep(; inner = nothing, nsteps = 10, adaptive = true, min_dλ = 1e-3)

Natural-parameter continuation solver for a [`SciMLBase.HomotopyProblem`](@ref). The
continuation parameter ``λ`` is swept across the problem's `λspan` in `nsteps` steps; each
step solves the inner nonlinear system with `inner`, warm-started from the previous step's
solution. When `adaptive` is `true`, a step whose inner solve fails to converge halves the
λ increment and retries, down to a floor of `min_dλ`.

`inner` is the inner nonlinear algorithm; `nothing` selects NonlinearSolve's default
polyalgorithm (NOT a hardcoded Newton). This is the embedding-homotopy / continuation
analogue used to robustly initialize systems whose target form is hard to solve cold; it is
unrelated to the polynomial `HomotopyContinuationJL`.
"""
@concrete struct HomotopySweep <: AbstractNonlinearSolveAlgorithm
    inner
    nsteps::Int
    adaptive::Bool
    min_dλ
end

function HomotopySweep(; inner = nothing, nsteps = 10, adaptive = true, min_dλ = 1.0e-3)
    return HomotopySweep(inner, nsteps, adaptive, min_dλ)
end
