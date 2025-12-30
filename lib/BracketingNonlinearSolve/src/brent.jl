"""
    Brent()

Left non-allocating Brent method.
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
    ϵ = eps(convert(typeof(fl), 1))

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
        @SciMLMessage("The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval)
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    if abs(fl) < abs(fr)
        left, right = right, left
        fl, fr = fr, fl
    end

    c = left
    d = c
    i = 1
    cond = true

    while i < maxiters
        fc = f(c)

        if fl != fc && fr != fc
            # Inverse quadratic interpolation
            s = left * fr * fc / ((fl - fr) * (fl - fc)) +
                right * fl * fc / ((fr - fl) * (fr - fc)) +
                c * fl * fr / ((fc - fl) * (fc - fr))
        else
            # Secant method
            s = right - fr * (right - left) / (fr - fl)
        end

        if (s < min((3 * left + right) / 4, right) ||
            s > max((3 * left + right) / 4, right)) ||
           (cond && abs(s - right) ≥ abs(right - c) / 2) ||
           (!cond && abs(s - right) ≥ abs(c - d) / 2) ||
           (cond && abs(right - c) ≤ ϵ) ||
           (!cond && abs(c - d) ≤ ϵ)
            # Bisection method
            s = (left + right) / 2
            if s == left || s == right
                return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.FloatingPointLimit)
            end
            cond = true
        else
            cond = false
        end

        fs = f(s)
        if abs((right - left) / 2) < abstol
            return build_bracketing_solution(prob, alg, s, fs, left, right, ReturnCode.Success)
        end

        if iszero(fs)
            if right < left
                left = right
                fl = fr
            end
            right = s
            fr = fs
            break
        end

        if fl * fs < 0
            d, c, right = c, right, s
            fr = fs
        else
            left = s
            fl = fs
        end

        if abs(fl) < abs(fr)
            d = c
            c, right = right, left
            left = c
            fc, fr = fr, fl
            fl = fc
        end
        i += 1
    end

    sol, i, left, right,
    fl, fr = Impl.bisection(
        left, right, fl, fr, f, abstol, maxiters - i, prob, alg
    )

    sol !== nothing && return sol

    return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.MaxIters)
end
