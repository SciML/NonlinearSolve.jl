module SCCNonlinearSolve

import SciMLBase
import CommonSolve
import SymbolicIndexingInterface
import SciMLBase: NonlinearProblem, NonlinearLeastSquaresProblem, LinearProblem

"""
    SCCAlg(; nlalg = nothing, linalg = nothing)

Algorithm for solving Strongly Connected Component (SCC) problems containing
both nonlinear and linear subproblems.

### Keyword Arguments

  - `nlalg`: Algorithm to use for solving NonlinearProblem components
  - `linalg`: Algorithm to use for solving LinearProblem components
"""
struct SCCAlg{N, L}
    nlalg::N
    linalg::L
end

SCCAlg(; nlalg = nothing, linalg = nothing) = SCCAlg(nlalg, linalg)

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(nothing, nothing); sensealg, u0, p, kwargs...)
end

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, ::Nothing; kwargs...)
    return CommonSolve.solve(prob; kwargs...)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem,
        alg::SciMLBase.AbstractNonlinearAlgorithm;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(alg, nothing); sensealg, u0, p, kwargs...)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    # Note: SCCNonlinearProblem does not have a u0 field - each subproblem has its own u0.
    # The u0 parameter here is only used for AD hooks, not for actual solving.
    p = p !== nothing ? p : prob.p
    return scc_solve_up(prob, sensealg, u0, p, alg; kwargs...)
end

"""
Internal solve function that can be hooked by ChainRulesCore for AD.
"""
function scc_solve_up(
        prob::SciMLBase.SCCNonlinearProblem, sensealg, u0, p, alg::SCCAlg;
        kwargs...
    )
    return _scc_solve(prob, alg; kwargs...)
end

probvec(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}) = prob.u0
probvec(prob::LinearProblem) = prob.b

iteratively_build_sols(alg, sols; kwargs...) = sols

function solve_single_scc(alg, prob, explicitfun; kwargs...)
    explicitfun(
        prob.p, sols
    )

    _sol = if prob isa SciMLBase.LinearProblem
        A = prob.A
        b = prob.b
        # `remake` to recalculate `A` and `b` based on updated parameters from `explicitfun`.
        # Pass `A` and `b` to avoid unnecessarily copying them.
        sol = SciMLBase.solve(SciMLBase.remake(prob; A, b), alg.linalg; kwargs...)
        # LinearSolution may have resid=nothing, so compute it: resid = A*u - b
        resid = isnothing(sol.resid) ? A * sol.u - b : sol.resid
        SciMLBase.build_linear_solution(
            alg.linalg, sol.u, resid, nothing, retcode = sol.retcode
        )
    else
        sol = SciMLBase.solve(prob, alg.nlalg; kwargs...)
        SciMLBase.build_solution(
            prob, nothing, sol.u, sol.resid, retcode = sol.retcode
        )
    end
    
    return _sol
end

function iteratively_build_sols(alg, probs::AbstractVector, explicitfuns::AbstractVector; kwargs...)
    solver = let alg = alg, probs = probs, explicitfuns = explicitfuns, kwargs = kwargs
        function _solver(i::Integer)
            return solve_single_scc(alg, probs[i], explicitfuns[i]; kwargs...)
        end
    end
    return map(solver, eachindex(probs))
end

@generated function iteratively_build_sols(alg, probs::Tuple, explicitfuns::Tuple, ::Val{N}; kwargs...) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> begin
            prob_i = solve_single_scc(alg, probs[i], explicitfuns[i]; kwargs...)
        end
        return Base.Cartesian.@ntuple $N i -> prob_i
    end
end

function iteratively_build_sols(alg, sols, (prob, explicitfun), args...; kwargs...)
    explicitfun(
        SymbolicIndexingInterface.parameter_values(prob), sols
    )

    _sol = if prob isa SciMLBase.LinearProblem
        A = prob.A
        b = prob.b
        # `remake` to recalculate `A` and `b` based on updated parameters from `explicitfun`.
        # Pass `A` and `b` to avoid unnecessarily copying them.
        sol = SciMLBase.solve(SciMLBase.remake(prob; A, b), alg.linalg; kwargs...)
        # LinearSolution may have resid=nothing, so compute it: resid = A*u - b
        resid = isnothing(sol.resid) ? A * sol.u - b : sol.resid
        SciMLBase.build_linear_solution(
            alg.linalg, sol.u, resid, nothing, retcode = sol.retcode
        )
    else
        sol = SciMLBase.solve(prob, alg.nlalg; kwargs...)
        SciMLBase.build_solution(
            prob, nothing, sol.u, sol.resid, retcode = sol.retcode
        )
    end

    return iteratively_build_sols(alg, (sols..., _sol), args...; kwargs...)
end

"""
Internal solve implementation for SCCNonlinearProblem.
This is called by scc_solve_up and should NOT be hooked by ChainRulesCore.
"""
function _scc_solve(prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg; kwargs...)
    numscc = length(prob.probs)
    if prob.probs isa Tuple
        sols = iteratively_build_sols(alg, prob.probs, prob.explicitfuns!, Val(numscc); kwargs...)
    else
        sols = iteratively_build_sols(alg, prob.probs, prob.explicitfuns!; kwargs...)
    end

    # TODO: fix allocations with a lazy concatenation
    u = reduce(vcat, sols)
    resid = reduce(vcat, getproperty.(sols, :resid))

    retcode = if !all(SciMLBase.successful_retcode, sols)
        idx = findfirst(!SciMLBase.successful_retcode, sols)
        sols[idx].retcode
    else
        SciMLBase.ReturnCode.Success
    end

    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end

export scc_solve_up

end
