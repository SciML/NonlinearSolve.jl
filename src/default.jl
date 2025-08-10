## TODO: In the long run we want to use an `Assumptions` API like LinearSolve to specify
##       the conditioning of the Jacobian and such

## TODO: Currently some of the algorithms like LineSearches / TrustRegion don't support
##       complex numbers. We should use the `DiffEqBase` trait for this once all of the
##       NonlinearSolve algorithms support it. For now we just do a check and remove the
##       unsupported ones from default

## Defaults to a fast and robust poly algorithm in most cases. If the user went through
## the trouble of specifying a custom jacobian function, we should use algorithms that
## can use that!
function SciMLBase.__init(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
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

function SciMLBase.__init(prob::SciMLBase.AbstractSteadyStateProblem, ::Nothing, args...; kwargs...)
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

function SciMLBase.__solve(prob::SciMLBase.AbstractSteadyStateProblem, ::Nothing, args...; kwargs...)
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

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem, ::Nothing, args...; kwargs...)
    return SciMLBase.__init(
        prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...; kwargs...
    )
end

function SciMLBase.__solve(
        prob::NonlinearLeastSquaresProblem, ::Nothing, args...; kwargs...
)
    return SciMLBase.__solve(
        prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...; kwargs...
    )
end

function NonlinearSolveBase.initialization_alg(::AbstractNonlinearProblem, autodiff)
    FastShortcutNonlinearPolyalg(; autodiff)
end
function NonlinearSolveBase.initialization_alg(::NonlinearLeastSquaresProblem, autodiff)
    FastShortcutNLLSPolyalg(; autodiff)
end
