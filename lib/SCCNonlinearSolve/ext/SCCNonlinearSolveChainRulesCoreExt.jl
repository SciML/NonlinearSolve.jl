module SCCNonlinearSolveChainRulesCoreExt

using SCCNonlinearSolve
using SCCNonlinearSolve: SCCAlg, scc_solve_up
using SciMLBase: SCCNonlinearProblem, AbstractSensitivityAlgorithm, ChainRulesOriginator,
    _concrete_solve_adjoint

import ChainRulesCore

function ChainRulesCore.rrule(
        ::typeof(scc_solve_up), prob::SCCNonlinearProblem,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, alg::SCCAlg; kwargs...)
    _concrete_solve_adjoint(prob, alg, sensealg, u0, p, ChainRulesOriginator(); kwargs...)
end

end
