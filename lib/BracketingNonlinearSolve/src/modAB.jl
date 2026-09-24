@inline same_signs(x, y) = (x < 0 && y < 0) || (x > 0 && y > 0)

@inline safe_midpoint(x1, x2) = x1 / 2 + x2 / 2

@inline function safe_secant(x1::T, y1, x2::T, y2) where {T <: AbstractFloat}
    a, b = abs(y1), abs(y2)
    den = a + b
    if isinf(den) # fast path
        (isinf(a) || isinf(b)) && return safe_midpoint(x1, x2)
        a /= 2
        b /= 2
        den = a + b
    end
    return clamp((b / den) * x1 + (a / den) * x2, x1, x2)
end

@inline function get_ab_factor(y3::T, y::T) where {T <: AbstractFloat}
    m = 1 - y3 / y
    return m > 0 ? m : inv(2 * one(m))
end

"""
    ModAB()

ModAB (Modified Anderson-Bjork)

Use the [ModAB method](https://iopscience.iop.org/article/10.1088/1757-899X/1276/1/012010/) to find a root of a bracketed
function, with a convergence rate between 1.7 and 1.8.

The method was introduced in the paper "Modified Anderson-Bjork's method for solving non-linear equations
in structural mechanics" (https://doi.org/10.1088/1757-899X/1276/1/012010) by
N Ganchovski and A Traykov.

This implementation includes the latest improvements made in 2026 by the following paper:
Ganchovski, N.; Smith, O.; Rackauckas, C.; Tomov, L.; Traykov, A.
Improvements to the Modified Anderson–Björck (modAB) Root-Finding Algorithm. Algorithms 2026, 19, 332.
(https://doi.org/10.3390/a19050332) and additional fixes by L.Tomov

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

    if same_signs(y1, y2)
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
    i = 1
    threshold = x2 - x1  # Threshold to fall back to bisection if AB fails to shrink the interval enough
    C = 2 # Safety factor for threshold corresponding to 2 iterations (2^2 * 0.5)
    f1, f2 = y1, y2 # The unmodified function values for correct calculation of symmetry factor after bisection fallback
    yMin = zero(y1) # The smallest unmodified residual of the bracket at the previous AB step
    while i < maxiters
        local x3, y3
        if bisecting # Bisection method is used
            x3 = x1/2 + x2/2 # Avoids possible overflow in x1 + x2
            y3 = f(x3) # Function value at midpoint
            if isfinite(f2 - f1)
                ym = (f1 + f2) / 2 # Ordinate of chord at midpoint
                r = 1 - abs(ym / (f2 - f1)) # Symmetry factor
                k = r * r # Deviation factor
                if abs(ym - y3) < k * abs(y3) + k * abs(ym) # Check if the function is close enough to linear
                    bisecting = false
                    threshold = C * (x2 - x1) # Initialize the bisection fallback threshold
                    y1, y2 = f1, f2  # A&B starts from the true residuals
                end
            end
        else # Anderson-Bjork method is used
            x3 = safe_secant(x1, y1, x2, y2)
            y3 = x3 == x1 ? f1 : x3 == x2 ? f2 : f(x3)
            threshold /= 2
            yMin = min(abs(f1), abs(f2))
        end
        if iszero(y3)
            return build_exact_solution(prob, alg, x3, y3, ReturnCode.Success)
        elseif isnan(y3)
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Failure)
        elseif (x2 - x1) < 2ϵ
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Success)
        end
        if bisecting
            if same_signs(f1, y3)
                x1, f1 = x3, y3
            else
                x2, f2 = x3, y3
            end
        else
            if same_signs(f1, y3)
                if side == 1  # Apply Anderson-Bjork correction on the right side
                    y2 *= get_ab_factor(y3, y1)
                else
                    side = 1
                end
                x1, y1, f1 = x3, y3, y3
            else
                if side == -1  # Apply Anderson-Bjork correction on the left side
                    y1 *= get_ab_factor(y3, y2)
                else
                    side = -1
                end
                x2, y2, f2 = x3, y3, y3
            end
            if x2 - x1 > threshold && abs(y3) > yMin / 2
                bisecting = true   # reset to bisection
                side = 0
            end
        end
        if nextfloat(x1) == x2
            return build_bracketing_solution(prob, alg, x2, f(x2), x1, x2, ReturnCode.FloatingPointLimit)
        end
        i += 1
    end
    return build_bracketing_solution(prob, alg, x1, f1, x1, x2, ReturnCode.MaxIters)
end
