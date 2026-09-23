@inline same_nonzero_sign(x, y) = (x < 0 && y < 0) || (x > 0 && y > 0)

@inline safe_midpoint(x1, x2) = x1 / 2 + x2 / 2

@inline function safe_secant(x1::T, y1, x2::T, y2) where {T <: AbstractFloat}
    a, b = abs(y1), abs(y2)
    den = a + b
    # den is always > 0 here: a NaN residual is rejected at the point of
    # evaluation, and neither residual is ever zero.
    if isinf(den)
        # An infinite ordinate carries no usable slope; otherwise a + b merely
        # overflowed, and halving both restores it without changing the ratio.
        # One halving always suffices: a, b <= floatmax implies a/2 + b/2 <= floatmax.
        (isinf(a) || isinf(b)) && return safe_midpoint(x1, x2)
        a /= 2
        b /= 2
        den = a + b
    end
    # Convex combination: the weights lie in [0, 1] and sum to 1, so this cannot
    # overflow for finite x1, x2. clamp only repairs last-ulp rounding.
    return clamp((b / den) * x1 + (a / den) * x2, x1, x2)
end

@inline function symmetry_factor(y1::T, y2::T) where {T <: AbstractFloat}
    a, b = abs(y1), abs(y2)
    den = a + b
    if isinf(den)
        # A non-finite residual disables switching; the switching test rejects
        # the NaN this produces.
        (isinf(a) || isinf(b)) && return T(NaN)
        a /= 2
        b /= 2
        den = a + b
    end
    r = 1 - abs(b - a) / den / 2
    return r * r
end

@inline function passes_switching_test(ym::T, yf::T, symmetry) where {T <: AbstractFloat}
    abs_ym, abs_yf = abs(ym), abs(yf)
    sum = abs_yf + abs_ym
    # Fast path. A non-finite ordinate or a NaN symmetry factor fails the
    # comparison and so disables switching, which is the intended behaviour.
    isfinite(sum) && return abs(ym - yf) < symmetry * sum
    # Only reached when the sum overflows: normalise the homogeneous inequality.
    isfinite(ym) && isfinite(yf) || return false
    scale = max(abs_yf, abs_ym)
    return abs(ym / scale - yf / scale) < symmetry * (abs_yf / scale + abs_ym / scale)
end

@inline function scale_preserving_nonzero_sign(value::T, positive_factor) where {T <: AbstractFloat}
    scaled = value * positive_factor
    (iszero(scaled) && !iszero(value)) && return copysign(nextfloat(zero(T)), value)   # double.Epsilon
    isinf(scaled) && return copysign(floatmax(T), value)          # double.MaxValue
    return scaled
end

@inline function ab_factor(y3::T, y::T) where {T <: AbstractFloat}
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

    if !((y1 < 0 && y2 > 0) || (y1 > 0 && y2 < 0))
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
    C = 2 # safety factor for threshold corresponding to 1 iteration = 2^1
    f1, f2 = y1, y2 # the unmodified function values for correct calculation of symmetry factor after bisection fallback
    yMin = zero(y1) # smallest unmodified residual of the bracket at the previous AB step
    while i < maxiters
        local x3, y3
        if bisecting # Bisection method is used
            x3 = safe_midpoint(x1, x2) # Avoids possible overflow in x1 + x2
            y3 = f(x3) # Function value at midpoint
            ym = safe_midpoint(f1, f2) # Ordinate of chord at midpoint
            # calculate k on each bisection step with account for local function properties and symmetry
            k = symmetry_factor(f1, f2)
            # Check if the function is close enough to linear
            if passes_switching_test(ym, y3, k)
                threshold = (x2 - x1) * C
                bisecting = false
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
        if same_nonzero_sign(y1, y3)
            if side == 1  # Apply Anderson-Bjork correction on the right side
                y2 = scale_preserving_nonzero_sign(y2, ab_factor(y3, y1))
            elseif !bisecting
                side = 1
            end
            x1, y1, f1 = x3, y3, y3
        else
            if side == -1  # Apply Anderson-Bjork correction on the left side
                y1 = scale_preserving_nonzero_sign(y1, ab_factor(y3, y2))
            elseif !bisecting
                side = -1
            end
            x2, y2, f2 = x3, y3, y3
        end
        if nextfloat(x1) == x2
            return build_bracketing_solution(prob, alg, x2, f(x2), x1, x2, ReturnCode.FloatingPointLimit)
        end
        i += 1
        if !bisecting && x2 - x1 > threshold && abs(y3) > yMin / 2
            bisecting = true   # reset to bisection
            side = 0
        end
    end
    return build_bracketing_solution(prob, alg, x1, y1, x1, x2, ReturnCode.MaxIters)
end
