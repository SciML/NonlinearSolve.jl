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

    # when using mooncake ∂sol would be a NamedTuple Tangent with cotangents of all the solution struct's fields.
    # However the pullback for this rule - "steadystatebackpass" as defined in SciMLSensitivity/src/concrete_solve.jl/
    # handles AD only when ∂sol is a ChainRulesCore.AbstractThunk object or a sol.u vector and similar data structures (not namedtuples).
    # When using Mooncake, we pass in sol.u to inner_thunking_pb directly as this is the only field relevant to the solution's cotangent (given solve_up, AbstractNonlinearProblem setting).

    function solve_up_adjoint(∂sol)
        adjoints = inner_thunking_pb(
            ∂sol isa Tangent{Any, <:NamedTuple} ? ∂sol.u : ∂sol
        )
        dp = adjoints[5]
        if p isa SciMLBase.DespecializedParameters && !(dp isa AbstractZero)
            dp = _unwrap_despecialized_tangent(dp)
            return Base.setindex(adjoints, Tangent{typeof(p)}(; params = dp), 5)
        end
        return adjoints
    end
    return primal, solve_up_adjoint
end

# Reverse-mode differentiation of a NonlinearLeastSquaresProblem solves the adjoint
# system of the (projected) stationarity equation, not the residual system: the
# residual is rectangular, and at active bounds it need not vanish at all. This is
# the reverse-mode counterpart of the ForwardDiff sensitivity handling in
# `NonlinearSolveBaseForwardDiffExt`, and shares its assumptions:
# parameter-independent bounds, a locally unchanged active set, and a nonsingular
# stationarity Jacobian on the free variables.
function SciMLBase._concrete_solve_adjoint(
        prob::SciMLBase.NonlinearLeastSquaresProblem, alg,
        sensealg::Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
        u0, p, originator::SciMLBase.ADOriginator,
        args...; save_idxs = nothing, kwargs...
    )
    if !(
            sensealg === nothing ||
                nameof(typeof(sensealg)) === :SteadyStateAdjoint
        )
        throw(
            ArgumentError(
                "NonlinearLeastSquaresProblem only supports the default \
                 stationarity adjoint (`SteadyStateAdjoint`). Got sensealg = \
                 $(nameof(typeof(sensealg)))."
            )
        )
    end
    p === nothing || p isa SciMLBase.NullParameters &&
        error(
        "Your model does not have parameters, and thus it is impossible to \
             calculate the derivative of the solution with respect to the \
             parameters."
    )
    p isa Union{Number, AbstractArray} || throw(
        ArgumentError(
            "Reverse-mode differentiation of a NonlinearLeastSquaresProblem \
             currently supports only scalar or array parameters; got \
             $(typeof(p))."
        )
    )

    _prob = remake(prob; u0, p)
    # `_prob` is already concrete, so call `solve_call` directly: re-entering
    # `solve`/`solve_up` re-runs `get_concrete_problem`, which re-wraps `f.f` in
    # `AutoSpecializeCallable`. On the Enzyme originator path `_prob.f.f` was
    # deliberately unwrapped to match the callable used in the traced forward
    # solve, and re-wrapping makes the returned `NonlinearSolution`'s type
    # disagree with the rule's declared primal type.
    kwargs_filtered = NamedTuple(filter(x -> x[1] != :originator, kwargs))
    sol = if alg isa NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
        NonlinearSolveBase.solve_call(_prob, alg, args...; kwargs_filtered...)
    else
        NonlinearSolveBase.solve_call(_prob, args...; kwargs_filtered...)
    end
    out = save_idxs === nothing ? sol :
        SciMLBase.sensitivity_solution(sol, sol.u[save_idxs])

    function nlls_solve_adjoint_backpass(Δ)
        Δ = Δ isa ChainRulesCore.AbstractThunk ? ChainRulesCore.unthunk(Δ) : Δ
        Δu = if Δ isa Union{Number, AbstractArray}
            Δ
        elseif Δ isa Tangent
            ChainRulesCore.backing(Δ).u
        elseif hasproperty(Δ, :u)
            Δ.u
        else
            Δ
        end
        Δu = Δu isa ChainRulesCore.AbstractThunk ? ChainRulesCore.unthunk(Δu) : Δu
        Δu isa ChainRulesCore.AbstractZero && (Δu = zero(sol.u))
        dp = NonlinearSolveBase.nlls_solve_adjoint_dp(_prob, sol, p, Δu, save_idxs)
        return if originator isa Union{
                SciMLBase.TrackerOriginator, SciMLBase.ReverseDiffOriginator,
            }
            (
                NoTangent(), NoTangent(), NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...,
            )
        else
            (
                NoTangent(), NoTangent(), NoTangent(),
                NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...,
            )
        end
    end
    return out, nlls_solve_adjoint_backpass
end

end
