"""
    Bisection(; exact_left = false, exact_right = false)

A common bisection method.

### Keyword Arguments

  - `exact_left`: whether to enforce whether the left side of the interval must be exactly
    zero for the returned result. Defaults to false.
  - `exact_right`: whether to enforce whether the right side of the interval must be exactly
    zero for the returned result. Defaults to false.

!!! danger "Keyword Arguments"

    Currently, the keyword arguments are not implemented.
"""
@kwdef struct Bisection <: AbstractBracketingAlgorithm
    exact_left::Bool = false
    exact_right::Bool = false
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Bisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
)
    @assert !SciMLBase.isinplace(prob) "`Bisection` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
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
        mid = (left + right) / 2

        if mid == left || mid == right
            return SciMLBase.build_solution(
                prob, alg, left, fl; retcode = ReturnCode.FloatingPointLimit, left, right
            )
        end

        fm = f(mid)
        if abs((right - left) / 2) < abstol
            return SciMLBase.build_solution(
                prob, alg, mid, fm; retcode = ReturnCode.Success, left, right
            )
        end

        if iszero(fm)
            right = mid
            break
        end

        if sign(fl) == sign(fm)
            fl = fm
            left = mid
        else
            fr = fm
            right = mid
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
