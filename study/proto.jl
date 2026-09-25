# Plain-function prototypes of derivative-using bracketing hybrids, for exploring variants
# before porting the promising ones into BracketingNonlinearSolve. Each method takes an
# oracle `o`, a bracket and an absolute tolerance, and returns `(u, left, right, code)`.
# The oracle counts the calls of f alone, of f' alone, of (f, f') together and of
# (f, f', f'') together.

mutable struct Oracle{F, D, FD, FDD}
    f::F        # x -> f(x)
    df::D       # x -> f'(x)
    fdf::FD     # x -> (f(x), f'(x)); a Dual pass when AD is used, else (f(x), df(x))
    fddf::FDD   # x -> (f(x), f'(x), f''(x))
    nf::Int
    nj::Int
    nfd::Int
    nfdd::Int
end
Oracle(f, df, fdf, fddf = nothing) = Oracle(f, df, fdf, fddf, 0, 0, 0, 0)
Oracle(f, df) = Oracle(f, df, x -> (f(x), df(x)))
fv(o, x) = (o.nf += 1; o.f(x))
dv(o, x) = (o.nj += 1; o.df(x))
fdv(o, x) = (o.nfd += 1; o.fdf(x))
fddv(o, x) = (o.nfdd += 1; o.fddf(x))

stopcode(x1, x2, tol) = nextfloat(x1) >= x2 ? :fpl : (x2 - x1) / 2 < tol ? :success : :continue
interior(t, x1, x2) = clamp(t, nextfloat(x1), prevfloat(x2))
better(x1, x2, f1, f2) = abs(f1) <= abs(f2) ? x1 : x2

# Root of the inverse cubic Hermite interpolant through (x1, f1, d1) and (x2, f2, d2).
function inverse_hermite(x1, x2, f1, f2, d1, d2)
    h = f2 - f1
    s = -f1 / h
    s2 = s * s
    s3 = s2 * s
    return (2s3 - 3s2 + 1) * x1 + (s3 - 2s2 + s) * h / d1 + (3s2 - 2s3) * x2 + (s3 - s2) * h / d2
end

"""
    hybrid(o, a, b, tol; model, push, C)

Model steps from the ends of the bracket, which carry f and f' (and f'' for Halley),
safeguarded by bisection: the bracket must halve per iteration on average, with a slack of
`C` (as ModAB's threshold), else the next point is the midpoint.

  - `model = :newton`: Newton from the end with the shorter Newton step.
  - `model = :hermite`: root of the inverse cubic Hermite interpolant through both ends when
    it is monotone, else Newton.
  - `model = :halley`: Halley from the end with the shorter Newton step.

End game (as NewtonBisection): once the step is within the tolerance, the point goes just
past the estimate on the other side of the root, doubling the distance until the sign
changes, so that both ends converge.

`push = true` also moves a Newton point that is predicted to land on the same side as its
base (from the sign of f''/f', with f'' from the derivatives at the two ends) past the root
by twice the predicted error, so that both ends converge before the end game.
"""
function hybrid(
        o, a, b, tol; model = :newton, push = false, C = 16, maxiters = 1000,
        state = nothing, safeguard = :rtsafe, multiplicity = false
    )
    halley = model === :halley
    if state === nothing
        x1, x2 = minmax(float.(promote(a, b))...)
        if halley
            f1, d1, h1 = fddv(o, x1)
            f2, d2, h2 = fddv(o, x2)
        else
            f1, d1 = fdv(o, x1)
            f2, d2 = fdv(o, x2)
            h1 = h2 = zero(f1)
        end
        iszero(f1) && return (x1, x1, x1, :exact)
        iszero(f2) && return (x2, x2, x2, :exact)
        sign(f1) == sign(f2) && return (x1, x1, x2, :fail)
    else
        x1, x2, f1, f2, d1, d2 = state
        h1 = h2 = zero(f1)
    end
    threshold = (x2 - x1) * C
    k = 1
    last_end = 0
    older = last = oftype(x2 - x1, Inf)
    last_base = 0
    last_δ = oftype(x2 - x1, Inf)
    for _ in 1:maxiters
        code = stopcode(x1, x2, tol)
        code === :continue || return (better(x1, x2, f1, f2), x1, x2, code)
        t = (x1 + x2) / 2
        model_step = false
        if safeguard === :rtsafe || x2 - x1 <= threshold
            s1, s2 = f1 / d1, f2 / d2
            use1 = !isfinite(s2) || (isfinite(s1) && abs(s1) <= abs(s2))
            xe, fe, de, he, se = use1 ? (x1, f1, d1, h1, s1) : (x2, f2, d2, h2, s2)
            est = xe - se
            if model === :hermite
                slope = (f2 - f1) / (x2 - x1)
                if d1 * slope > 0 && d2 * slope > 0
                    est = inverse_hermite(x1, x2, f1, f2, d1, d2)
                end
            elseif halley
                est = xe - 2fe * de / (2de^2 - fe * he)
            elseif multiplicity && last_base == (use1 ? 1 : 2) && abs(se) < last_δ
                # Newton near a root of multiplicity m shrinks the step by (m - 1)/m
                ρ = abs(se) / last_δ
                m = ρ > 1 // 2 ? min(round(inv(1 - ρ)), oftype(ρ, 16)) : one(ρ)
                est = xe - m * se
                isfinite(est) && x1 < est < x2 || (est = xe - se)
            end
            last_base, last_δ = use1 ? 1 : 2, abs(se)
            ok = safeguard === :rtsafe ? 2abs(est - xe) <= older : true
            δ = abs(est - xe)
            toward = xe == x1 ? 1 : -1    # direction from xe to the other end
            if isfinite(est) && δ <= max(tol, 16eps(xe))
                model_step = true
                Δ = k * max(tol, eps(xe))
                t = toward > 0 ? max(est, xe) + Δ : min(est, xe) - Δ
                t = interior(t, x1, x2)
                k *= 2
            elseif ok && isfinite(est) && x1 < est < x2
                model_step = true
                if push && model === :newton
                    K = (d2 - d1) / (x2 - x1) / (2de)   # N - root ≈ K δ²
                    # same side as xe when (N - root) has the sign of (xe - root) = -toward
                    if isfinite(K) && sign(K) == -toward
                        t = est + toward * min(2abs(K) * δ^2, δ / 2)
                    else
                        t = est
                    end
                else
                    t = est
                end
                t = interior(t, x1, x2)
            end
        end
        if halley
            ft, dt, ht = fddv(o, t)
        else
            ft, dt = fdv(o, t)
            ht = zero(ft)
        end
        older, last = last, abs(t - (abs(f1 / d1) <= abs(f2 / d2) ? x1 : x2))
        iszero(ft) && return (t, t, t, :exact)
        if sign(ft) == sign(f1)
            last_end == 2 && (k = 1)
            x1, f1, d1, h1 = t, ft, dt, ht
            last_end = 1
        else
            last_end == 1 && (k = 1)
            x2, f2, d2, h2 = t, ft, dt, ht
            last_end = 2
        end
        threshold = model_step ? threshold / 2 : (x2 - x1) * C
    end
    return (better(x1, x2, f1, f2), x1, x2, :maxiters)
end

"""
ModAB, unchanged, until it leaves its bisection phase (f looks linear on the bracket); then
the derivative at both ends and `hybrid` with the given model. Derivatives are only
evaluated in the second phase.
"""
function modab_polish(o, a, b, tol; model = :newton, push = false, maxiters = 1000, kernel = :hybrid, kw...)
    x1, x2 = minmax(float.(promote(a, b))...)
    y1, y2 = fv(o, x1), fv(o, x2)
    iszero(y1) && return (x1, x1, x1, :exact)
    iszero(y2) && return (x2, x2, x2, :exact)
    sign(y1) == sign(y2) && return (x1, x1, x2, :fail)
    for _ in 1:maxiters
        code = stopcode(x1, x2, tol)
        code === :continue || return (better(x1, x2, y1, y2), x1, x2, code)
        x3 = (x1 + x2) / 2
        y3 = fv(o, x3)
        iszero(y3) && return (x3, x3, x3, :exact)
        ym = (y1 + y2) / 2
        r = 1 - abs(ym / (y2 - y1))
        linear = abs(ym - y3) < r * r * (abs(ym) + abs(y3))
        if sign(y1) == sign(y3)
            x1, y1 = x3, y3
        else
            x2, y2 = x3, y3
        end
        if linear
            code = stopcode(x1, x2, tol)
            code === :continue || return (better(x1, x2, y1, y2), x1, x2, code)
            if kernel === :hybrid
                d1, d2 = dv(o, x1), dv(o, x2)
                return hybrid(o, a, b, tol; model, push, maxiters, state = (x1, x2, y1, y2, d1, d2), kw...)
            else
                n = oftype(y1, NaN)
                return newton_modab(o, a, b, tol; base = :smaller_f, lazy = true, push, maxiters, state = (x1, x2, y1, y2, n, n), modab = false, kw...)
            end
        end
    end
    return (better(x1, x2, y1, y2), x1, x2, :maxiters)
end

"""
Plain ModAB with a derivative-free Newton-like end game: none. Kept for reference: this is
the package ModAB, for checking the harness counts against `solve(prob, ModAB())`.
"""
function modab_ref(o, a, b, tol; maxiters = 1000)
    x1, x2 = minmax(float.(promote(a, b))...)
    y1, y2 = fv(o, x1), fv(o, x2)
    iszero(y1) && return (x1, x1, x1, :exact)
    iszero(y2) && return (x2, x2, x2, :exact)
    sign(y1) == sign(y2) && return (x1, x1, x2, :fail)
    bisecting = true
    side = 0
    threshold = x2 - x1
    for _ in 1:maxiters
        if bisecting
            x3 = (x1 + x2) / 2
            y3 = fv(o, x3)
            ym = (y1 + y2) / 2
            r = 1 - abs(ym / (y2 - y1))
            if abs(ym - y3) < r * r * (abs(ym) + abs(y3))
                bisecting = false
                threshold = (x2 - x1) * 16
            end
        else
            x3 = interior((x1 * y2 - y1 * x2) / (y2 - y1), x1, x2)
            y3 = fv(o, x3)
            threshold /= 2
        end
        iszero(y3) && return (x3, x3, x3, :exact)
        (x2 - x1) < 2tol && return (x3, x1, x2, :success)
        if sign(y1) == sign(y3)
            if side == 1
                m = 1 - y3 / y1
                y2 *= m <= 0 ? inv(2 * one(y1)) : m
            elseif !bisecting
                side = 1
            end
            x1, y1 = x3, y3
        else
            if side == -1
                m = 1 - y3 / y2
                y1 *= m <= 0 ? inv(2 * one(y1)) : m
            elseif !bisecting
                side = -1
            end
            x2, y2 = x3, y3
        end
        nextfloat(x1) == x2 && (fv(o, x2); return (x2, x1, x2, :fpl))
        if x2 - x1 > threshold
            bisecting = true
            side = 0
        end
    end
    return (x1, x1, x2, :maxiters)
end

"""
Newton from the better end of the bracket when that step is accepted, else the ModAB step
(bisection until f looks linear, then Anderson-Björck, back to bisection when the bracket
does not halve on average). Newton steps reset the Anderson-Björck weights.

  - `base = :shorter`: Newton from the end with the shorter Newton step (needs f' at both
    ends); `base = :smaller_f`: from the end with the smaller |f|.
  - `lazy = true`: evaluate f alone at each point and f' only at the Newton base, when it
    is not known yet (the natural mode for a user `jac`; with AD, f and f' come together and
    `lazy = false` is the natural mode).
  - `accept = :rtsafe`: Newton when its step is at most half the step before the previous
    one; `accept = :half`: when its step is at most half the bracket.
  - `multiplicity`: as in `hybrid`.
Same end game as `hybrid`.
"""
function newton_modab(
        o, a, b, tol; base = :shorter, lazy = false, accept = :rtsafe,
        multiplicity = false, push = false, maxiters = 1000, C = 16, state = nothing,
        modab = true
    )
    base === :shorter && (lazy = false)
    x1, x2 = minmax(float.(promote(a, b))...)
    if state !== nothing
        x1, x2, f1, f2, d1, d2 = state
    elseif lazy
        f1, f2 = fv(o, x1), fv(o, x2)
        d1 = d2 = oftype(f1, NaN)     # unknown
    else
        f1, d1 = fdv(o, x1)
        f2, d2 = fdv(o, x2)
    end
    iszero(f1) && return (x1, x1, x1, :exact)
    iszero(f2) && return (x2, x2, x2, :exact)
    sign(f1) == sign(f2) && return (x1, x1, x2, :fail)
    y1, y2 = f1, f2
    bisecting = true
    side = 0
    threshold = x2 - x1
    k = 1
    last_end = 0
    older = last = oftype(x2 - x1, Inf)
    last_base = 0
    last_δ = oftype(x2 - x1, Inf)
    for _ in 1:maxiters
        code = stopcode(x1, x2, tol)
        code === :continue || return (better(x1, x2, f1, f2), x1, x2, code)

        if base === :shorter
            s1, s2 = f1 / d1, f2 / d2
            use1 = !isfinite(s2) || (isfinite(s1) && abs(s1) <= abs(s2))
        else
            use1 = abs(f1) <= abs(f2)
        end
        if use1 && isnan(d1)
            d1 = dv(o, x1)
        elseif !use1 && isnan(d2)
            d2 = dv(o, x2)
        end
        xe, fe, de = use1 ? (x1, f1, d1) : (x2, f2, d2)
        se = fe / de
        est = xe - se
        if multiplicity && last_base == (use1 ? 1 : 2) && abs(se) < last_δ
            ρ = abs(se) / last_δ
            m = ρ > 1 // 2 ? min(round(inv(1 - ρ)), oftype(ρ, 16)) : one(ρ)
            est2 = xe - m * se
            isfinite(est2) && x1 < est2 < x2 && (est = est2)
        end
        last_base, last_δ = use1 ? 1 : 2, abs(se)
        δ = abs(est - xe)
        toward = use1 ? 1 : -1
        newton = true
        if isfinite(est) && δ <= max(tol, 16eps(xe))
            Δ = k * max(tol, eps(xe))
            t = interior(toward > 0 ? max(est, xe) + Δ : min(est, xe) - Δ, x1, x2)
            k *= 2
        elseif isfinite(est) && x1 < est < x2 &&
                (accept === :rtsafe ? 2δ <= older : 2δ <= x2 - x1)
            t = est
            if push && !isnan(d1) && !isnan(d2)
                K = (d2 - d1) / (x2 - x1) / (2de)
                if isfinite(K) && sign(K) == -toward
                    t = interior(est + toward * min(2abs(K) * δ^2, δ / 2), x1, x2)
                end
            end
        else
            newton = false
            t = bisecting || !modab ? (x1 + x2) / 2 :
                interior((x1 * y2 - y1 * x2) / (y2 - y1), x1, x2)
        end

        if lazy
            ft = fv(o, t)
            dt = oftype(ft, NaN)
        else
            ft, dt = fdv(o, t)
        end
        older, last = last, abs(t - xe)
        iszero(ft) && return (t, t, t, :exact)

        if newton
            side = 0
            threshold /= 2
        elseif bisecting
            ym = (y1 + y2) / 2
            r = 1 - abs(ym / (y2 - y1))
            if abs(ym - ft) < r * r * (abs(ym) + abs(ft))
                bisecting = false
                threshold = (x2 - x1) * C
            end
        else
            threshold /= 2
        end
        if sign(ft) == sign(f1)
            last_end == 2 && (k = 1)
            if !newton && side == 1
                m = 1 - ft / y1
                y2 *= m <= 0 ? inv(2 * one(y1)) : m
            elseif !newton && !bisecting
                side = 1
            end
            newton && (y2 = f2)
            x1, f1, y1, d1 = t, ft, ft, dt
            last_end = 1
        else
            last_end == 1 && (k = 1)
            if !newton && side == -1
                m = 1 - ft / y2
                y1 *= m <= 0 ? inv(2 * one(y1)) : m
            elseif !newton && !bisecting
                side = -1
            end
            newton && (y1 = f1)
            x2, f2, y2, d2 = t, ft, ft, dt
            last_end = 2
        end
        if x2 - x1 > threshold
            bisecting = true
            side = 0
            y1, y2 = f1, f2
        end
    end
    return (better(x1, x2, f1, f2), x1, x2, :maxiters)
end
