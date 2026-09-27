"""
    Brent()

Left non-allocating Brent method.

Brent's method as Brent (1973, *Algorithms for Minimization without Derivatives*,
ch. 4) and Numerical Recipes' `zbrent` give it: inverse quadratic interpolation or a
secant step when either shrinks the bracket fast enough, bisection otherwise, and a
minimum step -- when an interpolated step would move the estimate by less than
`tol1 = 2 eps(b) + abstol`, it moves by `tol1` towards the other end of the bracket
instead, so that the far end is replaced even when every interpolated iterate lands
on the same side of the root. Each iteration evaluates `f` once.
"""
struct Brent <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Brent, args...;
        maxiters = 1000, abstol = nothing, verbose = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`Brent` only supports out-of-place problems."

    if verbose isa Bool
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
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

    # `b` is the best estimate, `c` the contrapoint -- `f(b)` and `f(c)` have
    # opposite signs -- and `a` the previous `b`. `d` is the step just taken and `e`
    # the one before it. Every function value is carried with its point, so `f` is
    # evaluated once per iteration.
    a, fa = left, fl
    b, fb = right, fr
    c, fc = b, fb
    d = e = b - a

    for _ in 1:maxiters
        # keep the root between b and c
        if signbit(fb) == signbit(fc)
            c, fc = a, fa
            d = e = b - a
        end
        # keep b the better of the two
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end

        xm = (c - b) / 2
        if abs(xm) < abstol
            return build_bracketing_solution(prob, alg, b, fb, b, c, ReturnCode.Success)
        end
        if nextfloat(min(b, c)) == max(b, c)
            return build_bracketing_solution(
                prob, alg, b, fb, b, c, ReturnCode.FloatingPointLimit
            )
        end

        tol1 = 2 * eps(b) + abstol
        if abs(xm) <= tol1
            # The bracket is already at the resolution of the minimum step, which
            # only happens with an `abstol` below a few ulps: bisect down to
            # adjacent floats.
            d = xm
            e = d
        elseif abs(e) ≥ tol1 && abs(fa) > abs(fb)
            # interpolate: secant with two distinct points, else inverse quadratic
            s = fb / fa
            if a == c
                p = 2 * xm * s
                q = 1 - s
            else
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end
            p > 0 && (q = -q)
            p = abs(p)
            # accept it only if it stays inside the bracket and shrinks faster than
            # the step before last
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q))
                e = d
                d = p / q
            else
                d = xm
                e = d
            end
        else
            d = xm
            e = d
        end

        a, fa = b, fb
        # The minimum step: never move by less than tol1, and when the step would be
        # smaller, move by tol1 towards the contrapoint. Without it, once every
        # interpolated iterate lands on the same side of the root, the far end of the
        # bracket is only ever replaced by bisection. The minimum step is taken only
        # when `abs(xm) > tol1`, so the new point stays strictly inside the bracket.
        b += (abs(d) > tol1 || abs(xm) <= tol1) ? d : copysign(tol1, xm)
        fb = f(b)
        if iszero(fb)
            return build_exact_solution(prob, alg, b, fb, ReturnCode.Success)
        end
    end

    return build_bracketing_solution(prob, alg, b, fb, b, c, ReturnCode.MaxIters)
end
