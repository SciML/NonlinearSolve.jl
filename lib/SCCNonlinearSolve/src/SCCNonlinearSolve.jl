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

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem; kwargs...)
    CommonSolve.solve(prob, SCCAlg(nothing, nothing); kwargs...)
end

probvec(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}) = prob.u0
probvec(prob::LinearProblem) = prob.b

iteratively_build_sols(alg, sols; kwargs...) = sols

function iteratively_build_sols(alg, sols, (prob, explicitfun), args...; kwargs...)
    explicitfun(
        SymbolicIndexingInterface.parameter_values(prob), sols)

    _sol = if prob isa SciMLBase.LinearProblem
        sol = SciMLBase.solve(prob, alg.linalg; kwargs...)
        SciMLBase.build_linear_solution(
            alg.linalg, sol.u, nothing, nothing, retcode = sol.retcode)
    else
        sol = SciMLBase.solve(prob, alg.nlalg; kwargs...)
        SciMLBase.build_solution(
            prob, nothing, sol.u, sol.resid, retcode = sol.retcode)
    end

    iteratively_build_sols(alg, (sols..., _sol), args...)
end

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg; kwargs...)
    numscc = length(prob.probs)
    sols = iteratively_build_sols(
        alg, (), zip(prob.probs, prob.explicitfuns!)...; kwargs...)

    # TODO: fix allocations with a lazy concatenation
    u = reduce(vcat, sols)
    resid = reduce(vcat, getproperty.(sols, :resid))

    retcode = if !all(SciMLBase.successful_retcode, sols)
        idx = findfirst(!SciMLBase.successful_retcode, sols)
        sols[idx].retcode
    else
        SciMLBase.ReturnCode.Success
    end

    SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end

end
