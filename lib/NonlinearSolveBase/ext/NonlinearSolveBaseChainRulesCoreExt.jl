module NonlinearSolveBaseChainRulesCoreExt

using NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem, AutoSpecializeCallable,
    is_fw_wrapped, get_raw_f
using SciMLBase
using SciMLBase: AbstractSensitivityAlgorithm
using Setfield: @set

import ChainRulesCore
import ChainRulesCore: AbstractZero, NoTangent, Tangent

_unwrap_despecialized_tangent(dp) = SciMLBase.unwrap_parameters(dp)
_unwrap_despecialized_tangent(dp::NamedTuple{(:params,)}) = dp.params

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

    # `inner_thunking_pb` ("steadystatebackpass" in SciMLSensitivity/src/concrete_solve.jl)
    # consumes the state cotangent from `∂sol.u` and additionally the parameter cotangent
    # from `∂sol.prob.p` (populated by symbolic-indexing pullbacks such as `sol[obs]`).
    # Forwarding the whole tangent preserves both channels.

    function solve_up_adjoint(∂sol)
        adjoints = inner_thunking_pb(∂sol)
        dp = adjoints[5]
        if p isa SciMLBase.DespecializedParameters && !(dp isa AbstractZero)
            dp = _unwrap_despecialized_tangent(dp)
            return Base.setindex(adjoints, Tangent{typeof(p)}(; params = dp), 5)
        end
        return adjoints
    end
    return primal, solve_up_adjoint
end

end
