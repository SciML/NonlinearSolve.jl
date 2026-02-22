module NonlinearSolveBaseChainRulesCoreExt

using NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem, AutoSpecializeCallable,
    is_fw_wrapped, get_raw_f
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm
using Setfield: @set

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
    # Unwrap AutoSpecialize so that reverse-mode AD backends (Zygote, Mooncake, Enzyme)
    # never see FunctionWrapper types (whose llvmcall internals are not AD-compatible).
    if is_fw_wrapped(prob.f.f)
        prob = @set prob.f.f = get_raw_f(prob.f.f)
    end

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
