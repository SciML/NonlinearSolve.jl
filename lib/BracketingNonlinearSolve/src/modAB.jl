@inline same_signs(x, y) = (x < zero(x) && y < zero(y)) || (x > zero(x) && y > zero(y))

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

# Root of the linear-fractional model  f(x) = (x - r) / (p*x + q)  through the bracket
# (x1, y1), (x2, y2) and a third point (x3, f3) outside it. The bracket ordinates are the
# A&B-corrected residuals and f3 is the true residual of the dropped endpoint. The model fits
# both nearly-quadratic shapes and functions with a pole; for a straight line it gives the
# secant point. With w = x2 - x1, u = x3 - x1 and the residuals relative to y2,
# q1 = y1/y2 and q3 = f3/y2 (so the step does not depend on the scale or the units of f),
# and c = 1 - q1 > 1, the root is
#     r = x1 + t,   t = w*u*q1*(q3 - 1) / (w*(q3 - q1) - u*q3*c),
# two divisions per step, 1/y2 and this one. The A&B secant point is xs = x1 - w*q1/c and
# its distance to the nearer endpoint is d = w*min(|q1|, 1)/c. The step is kept within
# 0.7*d of xs (on a plateau the model degenerates towards a step function), so it stays
# at least 0.3*d inside the bracket. Multiplied by c, the test reads
#     |t*c + w*q1| <= 0.7*w*min(|q1|, 1),
# with no division. A step outside it is clamped to xs +- 0.7*d on the side the model
# points to; this did better than the A&B secant in randomized-bracket tests and on
# 200,000 random functions, where factors 0.6 to 0.9 differed little. The factor is
# written as 7/10 so that it keeps the type of the bracket (e.g. Float32).
# A step that rounds onto x2, or a NaN step (from overflow in t), returns x2. Its residual
# is already stored, so the caller reuses it instead of evaluating f; the closing step may
# first move the point to x2 - ϵ.
@inline function rational_step(x1, y1, x2, y2, x3, f3)
    k = inv(y2)
    q1, q3 = y1 * k, f3 * k
    c = 1 - q1
    w, u = x2 - x1, x3 - x1
    t = w * u * q1 * (q3 - 1) / (w * (q3 - q1) - u * q3 * c)
    a = abs(q1)
    lim = (a < one(a) ? w * a : w) * 7 / 10 # c * 0.7 * d
    dev = t * c + w * q1                    # c * (x - xs)
    # Clamp: t = ts +- 0.7*d. False for a NaN dev, which only comes from a NaN t (or q1, and
    # then t is NaN too), so x is NaN and the test below returns x2.
    if abs(dev) > lim
        t = ((dev < zero(dev) ? -lim : lim) - w * q1) / c
    end
    x = x1 + t
    # Rounding can put x on x2 when the bracket is a few ulps wide; NaN fails the test too.
    return x < x2 ? x : x2
end

# True if the second divided differences on the left (xl, x1, x2) and on the right
# (x1, x2, xr) have opposite signs: an inflection within the bracket, where a model with
# one-signed curvature does worse than the secant. Scaled by positive factors, they are
#     pl = (f2 - f1)*(x1 - xl) - (f1 - fl)*(x2 - x1),
#     pr = (fr - f2)*(x2 - x1) - (f2 - f1)*(xr - x2),
# so only their signs are compared, with no division or product of the two.
@inline function curvature_changes_sign(xl, fl, x1, f1, x2, f2, xr, fr)
    df, w = f2 - f1, x2 - x1
    pl = df * (x1 - xl) - (f1 - fl) * w
    pr = (fr - f2) * w - df * (xr - x2)
    return (pl < zero(pl) && pr > zero(pr)) || (pl > zero(pl) && pr < zero(pr))
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

The Anderson-Björck phase adds two changes:
1. Rational correction: the A&B secant is corrected for curvature by a 3-point rational model
   through the A&B-corrected bracket and the endpoint it most recently dropped, with the step
   kept within 0.7 of the distance from the A&B secant point to the nearer endpoint. The plain
   A&B secant is used when the curvature changes sign within the bracket.
2. Closing step: an A&B point is kept at least half the x-tolerance away from the endpoint
   that was just evaluated, so that one-sided convergence still closes the bracket.

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
    # Give ϵ the type of the bracket width; a zero tolerance also takes its units (e.g. a
    # plain `abstol = 0.0` with a DynamicQuantities bracket)
    ϵ = iszero(abstol) ? zero(x2 - x1) : oftype(x2 - x1, abstol)
    i = 1
    threshold = x2 - x1  # Threshold to fall back to bisection if AB fails to shrink the interval enough
    C = 2 # Safety factor for threshold corresponding to 2 iterations (2^2 * 0.5)
    f1, f2 = y1, y2 # The unmodified function values for correct calculation of symmetry factor after bisection fallback
    # A switching bisection whose midpoint reduces the residual of the replaced endpoint by less
    # than 1 % lies on a plateau; the first A&B step is then not rescaled.
    plateau_ratio = 0.99
    # Each return from A&B to bisection halves the margin of the switching test, so that
    # borderline switches are rejected after unsuccessful A&B phases.
    margin = one(abs(y1) / (abs(y1) + abs(y2)))
    last = 0 # Endpoint set by the last evaluation: 1 = x1, -1 = x2, 0 = none
    # Last endpoints dropped on the left and on the right, valid once has_l / has_r are set
    xl, fl, xr, fr = x1, f1, x2, f2
    has_l = has_r = false
    while i < maxiters
        local x3, y3
        switching = false # This bisection switches to A&B
        dx = x2 - x1 # Bracket width
        if bisecting # Bisection method is used
            x3 = safe_midpoint(x1, x2) # Avoids possible overflow in x1 + x2
            y3 = f(x3) # Function value at midpoint
        else # Anderson-Bjork method is used
            # Rational correction through the endpoint dropped by the last update, or the A&B
            # secant when the curvature changes sign within the bracket. has_l and has_r tell
            # which dropped endpoints exist; the curvature test needs both.
            if has_l && has_r && curvature_changes_sign(xl, fl, x1, f1, x2, f2, xr, fr)
                x3 = safe_secant(x1, y1, x2, y2)
            elseif last == 1 && has_l
                x3 = rational_step(x1, y1, x2, y2, xl, fl)
            elseif last == -1 && has_r
                x3 = rational_step(x1, y1, x2, y2, xr, fr)
            else
                x3 = safe_secant(x1, y1, x2, y2)
            end
            # Closing step: when A&B converges from one side, its next point lands within half
            # the x-tolerance (2ϵ) of the endpoint just evaluated and the bracket does not shrink.
            # A step of ϵ towards the far side then crosses the root and closes the bracket.
            if last == 1
                x3 - x1 < ϵ && x1 + ϵ < x2 && (x3 = x1 + ϵ)
            elseif x2 - x3 < ϵ && x2 - ϵ > x1
                x3 = x2 - ϵ
            end
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
                if abs(ym - y3) < (k * abs(y3) + k * abs(ym)) * margin # Check if the function is close enough to linear
                    bisecting = false
                    switching = true
                    threshold = C * dx # Initialize the bisection fallback threshold
                    y1, y2 = f1, f2  # A&B starts from the true residuals
                end
            end
        end
        if bisecting
            if same_signs(f1, y3)
                xl, fl, has_l = x1, f1, true
                x1, f1, last = x3, y3, 1
            else
                xr, fr, has_r = x2, f2, true
                x2, f2, last = x3, y3, -1
            end
        else # Anderson-Bjork method is used, including the step that switched to it
            yl, yr = f1, f2 # True residuals of the bracket before the update
            if same_signs(f1, y3)
                if side == 1  # Apply Anderson-Bjork correction on the right side
                    y2 *= get_ab_factor(y3, y1)
                elseif !switching || abs(y3) <= plateau_ratio * abs(yl)
                    side = 1 # A switching bisection on a plateau starts the A&B phase afresh
                end
                x3 != x1 && ((xl, fl, has_l) = (x1, f1, true))
                x1, y1, f1, last = x3, y3, y3, 1
            else
                if side == -1  # Apply Anderson-Bjork correction on the left side
                    y1 *= get_ab_factor(y3, y2)
                elseif !switching || abs(y3) <= plateau_ratio * abs(yr)
                    side = -1 # A switching bisection on a plateau starts the A&B phase afresh
                end
                x3 != x2 && ((xr, fr, has_r) = (x2, f2, true))
                x2, y2, f2, last = x3, y3, y3, -1
            end
            # Fallback if AB fails to reduce the bracket width, unless it still shrinks the residual more than 42 times
            # (2^(1074/200) = 41.4: at this rate the residual underflows within 200 steps from |y| = 1)
            if x2 - x1 > threshold && 42abs(y3) >= min(abs(yl), abs(yr))
                bisecting = true   # reset to bisection
                side = 0
                margin /= 2
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
