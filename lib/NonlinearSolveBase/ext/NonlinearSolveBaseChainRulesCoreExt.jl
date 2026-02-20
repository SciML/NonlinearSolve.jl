module NonlinearSolveBaseChainRulesCoreExt

using NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem, AutoSpecializeCallable,
    _DISABLE_AUTOSPECIALIZE
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm

import ChainRulesCore
import ChainRulesCore: NoTangent, Tangent

# Reverse-mode AD (Zygote, Mooncake) cannot differentiate through FunctionWrapper
# internals (llvmcall). When SciMLSensitivity builds the adjoint problem it calls
# the function through reverse-mode AD. This rule redirects differentiation through
# the original unwrapped callable, which is fully AD-compatible.
function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        f::AutoSpecializeCallable, args...
    )
    return ChainRulesCore.rrule_via_ad(config, f.orig, args...)
end

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
    # Disable AutoSpecialize wrapping for the entire adjoint code path.
    # Reverse-mode AD backends cannot differentiate through FunctionWrapper (llvmcall).
    # Mooncake in particular compiles tangent rules for ALL types in the forward pass,
    # so wrapping must be disabled before the forward solve runs.
    old = _DISABLE_AUTOSPECIALIZE[]
    _DISABLE_AUTOSPECIALIZE[] = true
    primal, inner_thunking_pb = try
        NonlinearSolveBase._solve_adjoint(
            prob, sensealg, u0, p,
            originator, args...;
            kwargs...
        )
    finally
        _DISABLE_AUTOSPECIALIZE[] = old
    end

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
