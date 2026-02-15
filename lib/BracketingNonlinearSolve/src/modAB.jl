"""
    ModAB(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)

ModAB (Interpolate Truncate & Project)

Use the [ModAB method](https://iopscience.iop.org/article/10.1088/1757-899X/1276/1/012010/) to find a root of a bracketed
function, with a convergence rate between 1 and 1.62.

This method was introduced in the paper "Modified Anderson-Bjork’s method for solving non-linear equations 
in structural mechanics" (https://doi.org/10.1088/1757-899X/1276/1/012010) by
N Ganchovski and A Traykov.
"""
struct ModAB <: AbstractBracketingAlgorithm
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::ModAB, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`ModAB` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    x1, x2 = minmax(promote(prob.tspan...)...)
    y1, y2 = f(x1), f(x2)

    abstol = NonlinearSolveBase.get_tolerance(
        x1, abstol, promote_type(eltype(x1), eltype(x2))
    )

    if iszero(y1)
        return build_exact_solution(prob, alg, x1, y1, ReturnCode.ExactSolutionLeft)
    end

    if iszero(y2)
        return build_exact_solution(prob, alg, x2, y2, ReturnCode.ExactSolutionRight)
    end

    if sign(y1) == sign(y2)
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, x1, y1, x1, x2, ReturnCode.InitialFailure)
    end
    
    bisecting = true
    side = 0 # tracks the side that has moved at the previous iteration
    ϵ = abstol
    N = -exponent(max(ϵ, nextfloat(zero(y1))))/2+1
    x0 = x1
    i = 1
    while i < maxiters
        local x3, y3
        if bisecting
            # Bisection
            x3 = (x1 + x2) / 2
            y3 = f(x3)
            # Ordinate of chord at midpoint
            ym = (y1 + y2) / 2
            if 4abs(ym - y3) < abs(ym) + abs(y3)
                bisecting = false
            end
        else
            # Falsi
            x3 = (x1*y2 - y1*x2) / (y2 - y1)
            y3 = f(x3)
        end

        if iszero(y3)
            return build_exact_solution(prob, alg, x3, y3, ReturnCode.Success)
        elseif abs(y3) < ϵ
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Success)
        end
        x0 = x3
        if side == 1
            m = 1 - y3/y1
            y2 *= m <= 0 ? inv(2*one(y1)) : m
        elseif side == 2 # Apply Anderson-Bjork modification for side 2
            m = 1 - y3/y2
            y1 *= m <= 0 ? inv(2*one(y1)) : m
        end
        if sign(y1) == sign(y3)
            if !bisecting
                side = 1
            end
            x1, y1 = x3, y3
        else
            if !bisecting
                side = 2
            end
            x2, y2 = x3, y3
        end
        if nextfloat(x1) == x2
            return build_bracketing_solution(prob, alg, x2, y2, x1, x2, ReturnCode.FloatingPointLimit)
        end
        i += 1
        if i >= N #taking longer than expected
            bisecting = true
            side = 0
            maxiters -= N
            i -= N
        end
    end

    return build_bracketing_solution(prob, alg, x1, y1, x1, x2, ReturnCode.MaxIters)
end
