module SCCNonlinearSolve

import SciMLBase
import CommonSolve
import SymbolicIndexingInterface

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

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg; kwargs...)
    numscc = length(prob.probs)
    sols = [SciMLBase.build_solution(
                prob, nothing, prob.u0, convert(eltype(prob.u0), NaN) * prob.u0)
            for prob in prob.probs]
    u = reduce(vcat, [prob.u0 for prob in prob.probs])
    resid = copy(u)

    lasti = 1
    for i in 1:numscc
        prob.explictfuns![i](
            SymbolicIndexingInterface.parameter_values(prob.probs[i]), sols)
        
        if prob.probs[i] isa SciMLBase.LinearProblem
            sol = SciMLBase.solve(prob.probs[i], alg.linalg; kwargs...)
            _sol = SciMLBase.build_solution(
                prob.probs[i], nothing, sol.u, zero(sol.u), retcode = sol.retcode)
        else
            sol = SciMLBase.solve(prob.probs[i], alg.nlalg; kwargs...)
            _sol = SciMLBase.build_solution(
                prob.probs[i], nothing, sol.u, sol.resid, retcode = sol.retcode)
        end
        
        sols[i] = _sol
        lasti = i
        if !SciMLBase.successful_retcode(_sol)
            break
        end
    end

    # TODO: fix allocations with a lazy concatenation
    u .= reduce(vcat, sols)
    resid .= reduce(vcat, getproperty.(sols, :resid))

    retcode = sols[lasti].retcode

    SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end


end
