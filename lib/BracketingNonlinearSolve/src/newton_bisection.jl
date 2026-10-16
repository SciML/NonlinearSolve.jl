"""
    NewtonBisection(; autodiff = nothing)

Newton's method safeguarded by bisection. The derivative of `f` is the `jac` of the
`IntervalNonlinearFunction` when it has one:

```julia
f(u, p) = u^2 - p
df(u, p) = 2u
prob = IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), (1.0, 2.0), 2.0)
sol = solve(prob, NewtonBisection())
```

Otherwise it comes from automatic differentiation with the `autodiff` backend, for example
`NewtonBisection(; autodiff = AutoForwardDiff())` after `using ForwardDiff`, which gives `f`
and its derivative together in one evaluation. Without `jac` and without `autodiff` it is
an error: with automatic differentiation the method is often slower than [`ModAB`](@ref),
see below.

Each iteration evaluates `f` once and then its derivative at the same point, so the two
can share work, for example through a cache of the last point. It takes the Newton step
from the last point, and bisects the bracket instead when that step leaves the bracket or
is not shorter than half the step before the previous one, as `rtsafe` of Numerical
Recipes does. Near a root of multiplicity `m`, where Newton's method converges only
linearly, successive steps from the same end shrink by a ratio `ρ ≈ (m - 1)/m`; when
`ρ > 1/2`, the step is multiplied by `m ≈ 1/(1 - ρ)` (at most 16) if that stays inside the
bracket. Once the Newton step is within the tolerance, the next point is placed just
beyond the Newton point, on the other side of the root, so that both ends of the bracket
converge.

Near a simple root the convergence is quadratic, and the method usually needs fewer
evaluations of `f` than the derivative-free methods. It evaluates the derivative about as
often as `f`, though, so it is faster only when the derivative costs a small fraction of
`f`: for example when `f` is an integral and the derivative its integrand, or when both
reuse the same expensive computation.
"""
@concrete struct NewtonBisection <: AbstractBracketingAlgorithm
    autodiff
end

function NewtonBisection(; autodiff::Union{Nothing, ADTypes.AbstractADType} = nothing)
    return NewtonBisection(autodiff)
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::NewtonBisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`NewtonBisection` only supports out-of-place problems."

    left, right = minmax(promote(prob.tspan...)...)
    return newton_bisection(
        prob, alg, interval_derivative(prob, alg, left), left, right;
        maxiters, abstol, verbose
    )
end

function newton_bisection(prob, alg, fun, left, right; maxiters, abstol, verbose)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    # The derivative is always evaluated right after `f`, at the same point.
    fl, dfl = value_then_derivative(fun, left)
    if iszero(fl)
        return build_exact_solution(prob, alg, left, fl, ReturnCode.ExactSolutionLeft)
    end
    dfl = derivative_after(fun, left, dfl)

    fr, dfr = value_then_derivative(fun, right)
    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, ReturnCode.ExactSolutionRight)
    end
    dfr = derivative_after(fun, right, dfr)

    if sign(fl) == sign(fr)
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    # Newton steps start from the end with the shorter step. The last point `x` is always
    # an end of the bracket.
    x, fx, dfx = abs(fl / dfl) <= abs(fr / dfr) ? (left, fl, dfl) : (right, fr, dfr)
    last_step = older_step = oftype(right - left, Inf)
    base_step = oftype(right - left, Inf)   # last plain Newton step from the same end
    base_left = x == left
    k = one(right - left)   # distance beyond the Newton point, in tolerances, once Newton has converged
    for _ in 1:maxiters
        if newton_bisection_done(left, right, abstol)
            retcode = nextfloat(left) == right ? ReturnCode.FloatingPointLimit : ReturnCode.Success
            return newton_bisection_solution(prob, alg, left, right, fl, fr, retcode)
        end

        step = fx / dfx
        newton = x - step
        if (x == left) == base_left && abs(step) < base_step
            ρ = abs(step) / base_step
            if 2ρ > 1
                # Newton shrinks the step by (m - 1)/m near a root of multiplicity m
                m = min(round(inv(1 - ρ)), oftype(ρ, 16))
                accelerated = x - m * step
                left < accelerated < right && (newton = accelerated)
            end
        end
        base_left, base_step = x == left, abs(step)
        δ = abs(newton - x)
        if isfinite(newton) && δ <= max(abstol, 16 * eps(x))
            # `x` is at the root up to the tolerance: close the other end of the bracket,
            # doubling the distance until the sign of `f` changes.
            δx = k * max(abstol, eps(x))
            t = x == left ? max(newton, x) + δx : min(newton, x) - δx
            t = clamp(t, nextfloat(left), prevfloat(right))
            k *= 2
        elseif left < newton < right && 2δ <= older_step
            t = newton
        else
            t = (left + right) / 2
        end
        older_step, last_step = last_step, abs(t - x)

        ft, dft = value_then_derivative(fun, t)
        iszero(ft) && return build_exact_solution(prob, alg, t, ft, ReturnCode.Success)
        was_left = x == left
        if sign(ft) == sign(fl)
            left, fl = t, ft
        else
            right, fr = t, ft
        end
        (t == left) == was_left || (k = one(k))
        x, fx = t, ft
        # The next step needs the derivative only if the bracket can still shrink.
        newton_bisection_done(left, right, abstol) || (dfx = derivative_after(fun, t, dft))
    end

    return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.MaxIters)
end

# `f` at `x`, and what `derivative_after` needs for `f'` at `x`: with automatic
# differentiation one pass gives both; a user `jac` is only called once it is needed.
value_then_derivative(d::UserDerivative, x) = (value(d, x), nothing)
value_then_derivative(d::ADDerivative, x) = value_and_derivative(d, x)
derivative_after(d::UserDerivative, x, ::Nothing) = derivative(d, x)
derivative_after(::ADDerivative, x, dfx) = dfx

# The bracket is within the tolerance, or its ends are adjacent floating-point numbers.
newton_bisection_done(left, right, abstol) = nextfloat(left) == right || (right - left) / 2 < abstol

# The end of the bracket with the smaller residual is the approximate root.
function newton_bisection_solution(prob, alg, left, right, fl, fr, retcode)
    u, fu = abs(fl) <= abs(fr) ? (left, fl) : (right, fr)
    return build_bracketing_solution(prob, alg, u, fu, left, right, retcode)
end
