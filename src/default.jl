## TODO: In the long run we want to use an `Assumptions` API like LinearSolve to specify
##       the conditioning of the Jacobian and such

## TODO: Currently some of the algorithms like LineSearches / TrustRegion don't support
##       complex numbers. We should use the `DiffEqBase` trait for this once all of the
##       NonlinearSolve algorithms support it. For now we just do a check and remove the
##       unsupported ones from default

## Defaults to a fast and robust poly algorithm in most cases. If the user went through
## the trouble of specifying a custom jacobian function, we should use algorithms that
## can use that!

## Bounded problems get the multistart-wrapped polyalgorithm: the first start is the
## user's u0, so a successful local solve costs exactly what the polyalgorithm costs;
## the deterministic restarts and restoration probes only run when the local solve
## stalls, where the alternative is reporting Stalled.
function _default_bounded_alg(prob, kwargs)
    return SobolMultistart(
        FastShortcutBoundedPolyalg(;
            must_support_postcondition = NonlinearSolveBase.get_postcondition(prob, kwargs) !== nothing
        )
    )
end

function SciMLBase.__init(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
    if prob.lb !== nothing || prob.ub !== nothing
        return SciMLBase.__init(prob, _default_bounded_alg(prob, kwargs), args...; kwargs...)
    end
    must_use_jacobian = Val(SciMLBase.has_jac(prob.f))
    return SciMLBase.__init(
        prob,
        FastShortcutNonlinearPolyalg(
            eltype(prob.u0); must_use_jacobian, u0_len = length(prob.u0)
        ),
        args...;
        kwargs...
    )
end

function SciMLBase.__solve(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
    if prob.lb !== nothing || prob.ub !== nothing
        return SciMLBase.__solve(prob, _default_bounded_alg(prob, kwargs), args...; kwargs...)
    end
    must_use_jacobian = Val(SciMLBase.has_jac(prob.f))
    prefer_simplenonlinearsolve = Val(prob.u0 isa StaticArray)
    return SciMLBase.__solve(
        prob,
        FastShortcutNonlinearPolyalg(
            eltype(prob.u0);
            must_use_jacobian,
            prefer_simplenonlinearsolve,
            u0_len = length(prob.u0)
        ),
        args...;
        kwargs...
    )
end

function __no_default_steadystate_alg_error(prob)
    throw(
        ArgumentError(
            "No default nonlinear algorithm exists for `$(typeof(prob).name.wrapper)` " *
                "here: `SciMLBase.NonlinearProblem` on it returns the problem itself " *
                "rather than a plain `NonlinearProblem` (e.g. an `SCCNonlinearProblem` " *
                "lowering), which this default-algorithm conversion cannot solve. " *
                "Specify an algorithm explicitly."
        )
    )
end

function SciMLBase.__init(prob::SciMLBase.AbstractSteadyStateProblem, ::Nothing, args...; kwargs...)
    # Convert SteadyStateProblem to NonlinearProblem and use its default
    nlprob = SciMLBase.NonlinearProblem(prob)
    # `NonlinearProblem(prob)` returns some inputs verbatim (a stored `NonlinearProblem`
    # or `LinearProblem` lowering, or — for `SCCNonlinearProblem` — `prob` itself, since
    # it has no `u0` field to rebuild from). Recursing on `nlprob === prob` would loop
    # forever instead of making progress toward a plain `NonlinearProblem`.
    nlprob === prob && __no_default_steadystate_alg_error(prob)
    return SciMLBase.__init(nlprob, nothing, args...; kwargs...)
end

function SciMLBase.__solve(prob::SciMLBase.AbstractSteadyStateProblem, ::Nothing, args...; kwargs...)
    # Convert SteadyStateProblem to NonlinearProblem and use its default
    nlprob = SciMLBase.NonlinearProblem(prob)
    nlprob === prob && __no_default_steadystate_alg_error(prob)
    return SciMLBase.__solve(nlprob, nothing, args...; kwargs...)
end

function NonlinearSolveBase.initialization_alg(prob::AbstractNonlinearProblem, autodiff)
    if prob isa NonlinearProblem && (prob.lb !== nothing || prob.ub !== nothing)
        return SobolMultistart(
            FastShortcutBoundedPolyalg(;
                autodiff,
                must_support_postcondition = NonlinearSolveBase.get_postcondition(prob, (;)) !== nothing
            )
        )
    end
    return FastShortcutNonlinearPolyalg(; autodiff)
end

# A `HomotopyProblem` initialization system (e.g. a Modelica `homotopy` operator) must be
# continued, not solved at the target `λ`: the `AbstractNonlinearProblem` method above would
# hand it a plain nonlinear polyalgorithm that fixes `λ` and solves only the `actual` system
# (see `solve(::HomotopyProblem, ::AbstractNonlinearSolveAlgorithm)`), which can land on the
# wrong branch. Route it to the continuation default instead, carrying the same `autodiff`.
function NonlinearSolveBase.initialization_alg(::SciMLBase.HomotopyProblem, autodiff)
    return FastShortcutHomotopyPolyalg(; autodiff)
end
