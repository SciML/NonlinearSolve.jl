module SCCNonlinearSolve

import SciMLBase
import CommonSolve
import SymbolicIndexingInterface
import SciMLBase: NonlinearProblem, NonlinearLeastSquaresProblem, LinearProblem

"""
    SCCAlg(; nlalg = nothing, linalg = nothing, store_original = Val(false))

Algorithm for solving Strongly Connected Component (SCC) problems containing
both nonlinear and linear subproblems.

### Keyword Arguments

  - `nlalg`: Algorithm to use for solving NonlinearProblem components
  - `linalg`: Algorithm to use for solving LinearProblem components
  - `store_original`: Whether to store the individual sub-solutions in the
    `original` field of the returned solution. Default `Val(false)` to keep
    the return type simple (required for Enzyme AD compatibility). Set to
    `Val(true)` for debugging to inspect individual sub-solutions.
"""
struct SCCAlg{N, L, S <: Val}
    nlalg::N
    linalg::L
    store_original::S
end

function SCCAlg(; nlalg = nothing, linalg = nothing, store_original = Val(false))
    return SCCAlg(nlalg, linalg, store_original)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(); sensealg, u0, p, kwargs...)
end

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, ::Nothing; kwargs...)
    return CommonSolve.solve(prob; kwargs...)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem,
        alg::SciMLBase.AbstractNonlinearAlgorithm;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(; nlalg = alg); sensealg, u0, p, kwargs...)
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

function solve_single_scc(alg, prob, explicitfun, sols; kwargs...)
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
        nlprob = NonlinearProblem{true}(Returns(nothing), sol.u, prob.p)
        SciMLBase.strip_solution(
            SciMLBase.build_solution(nlprob, nothing, sol.u, resid, retcode = sol.retcode)
        )
    else
        sol = SciMLBase.solve(prob, alg.nlalg; kwargs...)
        SciMLBase.strip_solution(
            SciMLBase.build_solution(
                prob, nothing, sol.u, sol.resid, retcode = sol.retcode
            )
        )
    end

    return _sol
end

function iteratively_build_sols(alg, probs::AbstractVector, explicitfuns::AbstractVector; kwargs...)
    # Compute the stripped solution type deterministically from the first problem.
    # After strip_solution, all NonlinearSolutions have a predictable concrete type
    # regardless of the original problem/algorithm types.
    prob1 = first(probs)
    uType = typeof(probvec(prob1))
    T = eltype(uType)
    rType = uType  # resid has same type as u for nonlinear problems
    ST = SciMLBase.NonlinearSolution{
        T, 1, uType, rType,
        NamedTuple{(:p,), Tuple{Nothing}}, Nothing, Nothing, Nothing, Nothing, Nothing,
    }
    sols = Vector{ST}(undef, length(probs))
    for i in eachindex(probs)
        sols[i] = solve_single_scc(alg, probs[i], explicitfuns[i], view(sols, 1:(i - 1)); kwargs...)::ST
    end
    return sols
end

@generated function iteratively_build_sols(alg, probs::Tuple, explicitfuns::Tuple, ::Val{N}; kwargs...) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> begin
            prob_i = solve_single_scc(
                alg, probs[i], explicitfuns[i],
                Base.Cartesian.@ntuple((i - 1), j -> prob_j); kwargs...
            )
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
        sols = iteratively_build_sols(alg, prob.probs, Tuple(prob.explicitfuns!), Val(numscc); kwargs...)
    else
        sols = iteratively_build_sols(alg, prob.probs, prob.explicitfuns!; kwargs...)
    end

    # Use splatted vcat instead of reduce (reduce over tuples infers Any,
    # which exceeds Enzyme's type analysis limits).
    u = vcat(map(s -> s.u, sols)...)
    resid = vcat(map(s -> s.resid, sols)...)

    # Simple loop instead of all/findfirst (higher-order functions over
    # tuples can confuse Enzyme's activity analysis).
    retcode = SciMLBase.ReturnCode.Success
    for s in sols
        if !SciMLBase.successful_retcode(s)
            retcode = s.retcode
            break
        end
    end

    # Only store sub-solutions in `original` when requested (Val(true)).
    # The sub-solutions tuple nested in the kwargs NamedTuple exceeds
    # Enzyme's type analysis depth, so it's off by default.
    original = alg.store_original isa Val{true} ? sols : nothing
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original)
end

export scc_solve_up

end
