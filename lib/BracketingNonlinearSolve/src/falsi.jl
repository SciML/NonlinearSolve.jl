"""
    Falsi()

A non-allocating regula falsi method.
"""
struct Falsi <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Falsi, args...;
        maxiters = 1000, abstol = nothing, verbose = NonlinearVerbosity(), kwargs...
)
    @assert !SciMLBase.isinplace(prob) "`False` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    l, r = prob.tspan # don't reuse these variables
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right)))

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right
        )
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right
        )
    end

    if sign(fl) == sign(fr)
        verbose &&
            @warn "The interval is not an enclosing interval, opposite signs at the \
                   boundaries are required."
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right
        )
    end

    i = 1
    while i ≤ maxiters
        if Impl.nextfloat_tdir(left, l, r) == right
            return SciMLBase.build_solution(
                prob, alg, left, fl; left, right, retcode = ReturnCode.FloatingPointLimit
            )
        end

        mid = (fr * left - fl * right) / (fr - fl)
        for _ in 1:10
            mid = Impl.max_tdir(left, Impl.prevfloat_tdir(mid, l, r), l, r)
        end

        (mid == left || mid == right) && break

        fm = f(mid)
        if abs((right - left) / 2) < abstol
            return SciMLBase.build_solution(
                prob, alg, mid, fm; left, right, retcode = ReturnCode.Success
            )
        end

        if abs(fm) < abstol
            right = mid
            break
        end

        if sign(fl) == sign(fm)
            fl, left = fm, mid
        else
            fr, right = fm, mid
        end

        i += 1
    end

    sol, i, left, right,
    fl, fr = Impl.bisection(
        left, right, fl, fr, f, abstol, maxiters - i, prob, alg
    )

    sol !== nothing && return sol

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right
    )
end
