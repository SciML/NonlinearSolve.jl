@inline same_signs(x, y) = (x < 0 && y < 0) || (x > 0 && y > 0)

@inline safe_midpoint(x1, x2) = x1 / 2 + x2 / 2

@inline function safe_secant(x1, y1, x2, y2)
    a, b = abs(y1), abs(y2)
    den = a + b
    if isinf(den) # fast path
        (isinf(a) || isinf(b)) && return safe_midpoint(x1, x2)
        a /= 2
        b /= 2
        den = a + b
    end
    return (b / den) * x1 + (a / den) * x2 # May round outside [x1, x2]; the caller handles that
end

@inline function get_ab_factor(y3, y)
    m = 1 - y3 / y
    return m > 0 ? m : inv(2one(m))
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

If `f` evaluates to `NaN` at an iterate, the solver stops and returns
`ReturnCode.Failure` with that iterate and the current bracket.
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

    # The secant step mixes abscissae with residual ratios, so promote the bracket to that
    # type once (e.g. Float32 tspan with Float64 residuals, or Dual residuals from a
    # closure-captured Dual) to keep x1 and x2 the same type throughout the iterations.
    T = typeof(abs(y1) / (abs(y1) + abs(y2)) * x1)
    x1, x2 = convert(T, x1), convert(T, x2)

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
    MaxResidualSteps = 3 # Max consecutive AB steps kept by the residual test alone
    residualSteps = 0
    while i < maxiters
        local x3, y3
        dx = x2 - x1 # Bracket width
        if bisecting # Bisection method is used
            x3 = safe_midpoint(x1, x2) # Avoids possible overflow in x1 + x2
            y3 = f(x3) # Function value at midpoint
        else # Anderson-Bjork method is used
            x3 = safe_secant(x1, y1, x2, y2)
            if x3 <= x1 # Rounded onto or past an endpoint: reuse its true residual
                x3, y3 = x1, f1
            elseif x3 >= x2
                x3, y3 = x2, f2
            else
                y3 = f(x3)
            end
        end
        if iszero(y3)
            return build_exact_solution(prob, alg, x3, y3, ReturnCode.Success)
        elseif isnan(y3)
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Failure)
        elseif dx < 2ϵ
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Success)
        end
        if bisecting
            dy = f2 - f1
            if isfinite(dy)
                ym = (f1 + f2) / 2 # Ordinate of chord at midpoint
                r = 1 - abs(ym / dy) # Symmetry factor
                k = r * r # Deviation factor
                if abs(ym - y3) < k * abs(y3) + k * abs(ym) # Check if the function is close enough to linear
                    bisecting = false
                    threshold = C * dx # Initialize the bisection fallback threshold
                    residualSteps = 0
                    y1, y2 = f1, f2  # A&B starts from the true residuals
                end
            end
        end
        if bisecting
            if same_signs(f1, y3)
                x1, f1 = x3, y3
            else
                x2, f2 = x3, y3
            end
        else # Anderson-Bjork method is used, including the step that switched to it
            yl, yr = f1, f2 # True residuals of the bracket before the update
            if same_signs(f1, y3)
                if side == 1  # Apply Anderson-Bjork correction on the right side
                    y2 *= get_ab_factor(y3, y1)
                end
                x1, y1, f1, side = x3, y3, y3, 1
            else
                if side == -1  # Apply Anderson-Bjork correction on the left side
                    y1 *= get_ab_factor(y3, y2)
                end
                x2, y2, f2, side = x3, y3, y3, -1
            end
            # Fallback if AB fails to reduce the bracket width, unless it still halves the residual,
            # but for no more than MaxResidualSteps consecutive steps
            if x2 - x1 > threshold
                yMin = min(abs(yl), abs(yr)) # Best true residual of the bracket
                if residualSteps >= MaxResidualSteps || 2abs(y3) >= yMin
                    bisecting = true   # reset to bisection
                    side = 0
                else
                    residualSteps += 1
                end
            else
                residualSteps = 0
            end
            threshold /= 2
        end
        if nextfloat(x1) == x2
            return build_bracketing_solution(prob, alg, x2, f2, x1, x2, ReturnCode.FloatingPointLimit)
        end
        i += 1
    end
    return build_bracketing_solution(prob, alg, x1, f1, x1, x2, ReturnCode.MaxIters)
end
