module NonlinearSolveBaseChainRulesCoreExt

using NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm

import ChainRulesCore
import ChainRulesCore: NoTangent

function ChainRulesCore.frule(::typeof(NonlinearSolveBase.solve_up), prob,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...)
    NonlinearSolveBase._solve_forward(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...)
end

function ChainRulesCore.rrule(::typeof(NonlinearSolveBase.solve_up), prob::AbstractNonlinearProblem,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...)
    NonlinearSolveBase._solve_adjoint(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...)
end

end