module NonlinearSolveBaseChainRulesCoreExt

using NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm

import ChainRulesCore
import ChainRulesCore: NoTangent, Tangent

function ChainRulesCore.frule(
        ::typeof(NonlinearSolveBase.solve_up), prob,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    return NonlinearSolveBase._solve_forward(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...
    )
end

function ChainRulesCore.rrule(
        ::typeof(NonlinearSolveBase.solve_up), prob::AbstractNonlinearProblem,
        sensealg::Union{Nothing, AbstractSensitivityAlgorithm},
        u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    primal, inner_thunking_pb = NonlinearSolveBase._solve_adjoint(
        prob, sensealg, u0, p,
        originator, args...;
        kwargs...
    )

    # when using mooncake ∂sol would be a NamedTuple Tangent with cotangents of all the solution struct's fields.
    # However the pullback for this rule - "steadystatebackpass" as defined in SciMLSensitivity/src/concrete_solve.jl/
    # handles AD only when ∂sol is a ChainRulesCore.AbstractThunk object or a sol.u vector and similar data structures (not namedtuples).
    # When using Mooncake, we pass in sol.u to inner_thunking_pb directly as this is the only field relevant to the solution's cotangent (given solve_up, AbstractNonlinearProblem setting).

    function solve_up_adjoint(∂sol)
        return inner_thunking_pb(∂sol isa Tangent{Any, <:NamedTuple} ? ∂sol.u : ∂sol)
    end
    return primal, solve_up_adjoint
end

end
