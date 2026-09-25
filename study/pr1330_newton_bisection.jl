# The NewtonBisection of https://github.com/SciML/NonlinearSolve.jl/pull/1330 (commit a84b9c59),
# unchanged apart from its name, as the reference for the benchmarks. Uses internals of
# BracketingNonlinearSolve; study code only.
using BracketingNonlinearSolve: BracketingNonlinearSolve, AbstractBracketingAlgorithm,
    build_exact_solution, build_bracketing_solution
using NonlinearSolveBase: NonlinearSolveBase, NonlinearVerbosity
using SciMLBase: SciMLBase, IntervalNonlinearProblem, ReturnCode
using SciMLLogging: @SciMLMessage

struct PRNewtonBisection <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::PRNewtonBisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`PRNewtonBisection` only supports out-of-place problems."
    SciMLBase.has_jac(prob.f) || throw(
        ArgumentError(
            "`PRNewtonBisection` needs the derivative of `f`, given as \
            `IntervalNonlinearFunction(f; jac)`."
        )
    )

    f = Base.Fix2(prob.f, prob.p)
    df = Base.Fix2(prob.f.jac, prob.p)
    left, right = minmax(promote(prob.tspan...)...)
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    if iszero(fl)
        return build_exact_solution(prob, alg, left, fl, ReturnCode.ExactSolutionLeft)
    end

    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, ReturnCode.ExactSolutionRight)
    end

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
    dfl, dfr = df(left), df(right)
    x, fx, dfx = abs(fl / dfl) <= abs(fr / dfr) ? (left, fl, dfl) : (right, fr, dfr)
    last_step = older_step = oftype(right - left, Inf)
    k = 1   # distance beyond the Newton point, in tolerances, once Newton has converged
    for _ in 1:maxiters
        if nextfloat(left) == right
            return pr_newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.FloatingPointLimit)
        end
        if (right - left) / 2 < abstol
            return pr_newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.Success)
        end

        newton = x - fx / dfx
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

        ft = f(t)
        iszero(ft) && return build_exact_solution(prob, alg, t, ft, ReturnCode.Success)
        was_left = x == left
        if sign(ft) == sign(fl)
            left, fl = t, ft
        else
            right, fr = t, ft
        end
        (t == left) == was_left || (k = 1)
        x, fx, dfx = t, ft, df(t)
    end

    return pr_newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.MaxIters)
end

# The end of the bracket with the smaller residual is the approximate root.
function pr_newton_bisection_solution(prob, alg, left, right, fl, fr, retcode)
    u, fu = abs(fl) <= abs(fr) ? (left, fl) : (right, fr)
    return build_bracketing_solution(prob, alg, u, fu, left, right, retcode)
end
