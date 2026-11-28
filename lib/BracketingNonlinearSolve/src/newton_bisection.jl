"""
    NewtonBisection()

Newton's method safeguarded by bisection, for problems that give the derivative of `f` as
the `jac` of their `IntervalNonlinearFunction`:

```julia
f(u, p) = u^2 - p
df(u, p) = 2u
prob = IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), (1.0, 2.0), 2.0)
sol = solve(prob, NewtonBisection())
```

Each iteration evaluates `f` once and then its derivative at the same point, so the two
can share work, for example through a cache of the last point. It takes the Newton step
from the last point, and bisects the bracket instead when that step leaves the bracket or
is not shorter than half the step before the previous one, as `rtsafe` of Numerical
Recipes does. Once the Newton step is within the tolerance, the next point is placed just
beyond the Newton point, on the other side of the root, so that both ends of the bracket
converge.

It stops when the bracket is narrower than `abstol`, or its ends are adjacent floats, and
returns the end with the smaller residual, so `sol.u` is within `abstol` of the root.

Near a simple root the convergence is quadratic, and the method usually needs fewer
evaluations of `f` than the derivative-free methods. It evaluates the derivative about as
often as `f`, though, so it is faster only when the derivative costs a small fraction of
`f`: for example when `f` is an integral and the derivative its integrand, or when both
reuse the same expensive computation.

If `f` evaluates to `NaN`, at an end of the bracket or inside it, the solver stops and
returns `ReturnCode.Failure` with that point and the current bracket. A derivative that is
zero, infinite or `NaN` only makes the method bisect.
"""
struct NewtonBisection <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::NewtonBisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`NewtonBisection` only supports out-of-place problems."
    SciMLBase.has_jac(prob.f) || throw(
        ArgumentError(
            "`NewtonBisection` needs the derivative of `f`, given as \
            `IntervalNonlinearFunction(f; jac)`."
        )
    )

    f = Base.Fix2(prob.f, prob.p)
    df = Base.Fix2(prob.f.jac, prob.p)
    left, right = minmax(promote(prob.tspan...)...)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    # The exact-solution codes follow the order of `tspan`, not of the sorted bracket.
    exact = (ReturnCode.ExactSolutionLeft, ReturnCode.ExactSolutionRight)
    exact_left, exact_right = prob.tspan[1] <= prob.tspan[2] ? exact : reverse(exact)

    # The derivative is always evaluated right after `f`, at the same point.
    # A `NaN` residual has no sign, so it can't tell on which side of the root it lies.
    fl = f(left)
    # The bracket takes the residual type if wider, as in `ModAB`, so it is never rounded.
    T = float(promote_type(typeof(left), typeof(fl)))
    left, right = convert(T, left), convert(T, right)
    if iszero(fl)
        return build_exact_solution(prob, alg, left, fl, exact_left)
    end
    isnan(fl) && return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.Failure)
    dfl = df(left)

    fr = f(right)
    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, exact_right)
    end
    isnan(fr) && return build_bracketing_solution(prob, alg, right, fr, left, right, ReturnCode.Failure)
    dfr = df(right)

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
    tol = abstol / 2   # so that the bracket closed by Newton is narrower than `abstol`
    k = one(right - left)   # distance beyond the Newton point, in tolerances, once Newton has converged
    newton_bisection_done(left, right, abstol) &&
        return newton_bisection_solution(prob, alg, left, right, fl, fr, abstol)
    for _ in 1:maxiters
        newton = x - fx / dfx
        δ = abs(newton - x)
        # An infinite derivative gives a zero step, which is not convergence.
        if isfinite(dfx) && δ <= max(tol, 16 * eps(x))
            # `x` is at the root up to the tolerance: close the other end of the bracket,
            # doubling the distance until the sign of `f` changes.
            δx = k * max(tol, eps(x))
            t = x == left ? max(newton, x) + δx : min(newton, x) - δx
            k *= 2
        elseif left < newton < right && 2δ <= older_step
            t = newton
        else
            t = safe_midpoint(left, right)
        end
        t = clamp(convert(T, t), nextfloat(left), prevfloat(right))
        older_step, last_step = last_step, abs(t - x)

        ft = f(t)
        iszero(ft) && return build_exact_solution(prob, alg, t, ft, ReturnCode.Success)
        isnan(ft) && return build_bracketing_solution(prob, alg, t, ft, left, right, ReturnCode.Failure)
        was_left = x == left
        if sign(ft) == sign(fl)
            left, fl = t, ft
        else
            right, fr = t, ft
        end
        (t == left) == was_left || (k = one(k))
        x, fx = t, ft
        # The next step needs the derivative only if the bracket can still shrink.
        newton_bisection_done(left, right, abstol) &&
            return newton_bisection_solution(prob, alg, left, right, fl, fr, abstol)
        dfx = df(t)
    end

    return newton_bisection_solution(prob, alg, left, right, fl, fr, abstol)
end

# The bracket is within the tolerance, or its ends are adjacent floating-point numbers.
newton_bisection_done(left, right, abstol) = nextfloat(left) == right || right - left < abstol

# The end of the bracket with the smaller residual is the approximate root.
function newton_bisection_solution(prob, alg, left, right, fl, fr, abstol)
    retcode = nextfloat(left) == right ? ReturnCode.FloatingPointLimit :
        right - left < abstol ? ReturnCode.Success : ReturnCode.MaxIters
    u, fu = abs(fl) <= abs(fr) ? (left, fl) : (right, fr)
    return build_bracketing_solution(prob, alg, u, fu, left, right, retcode)
end
