"""
    Brent()

Left non-allocating Brent method.
"""
struct Brent <: AbstractBracketingAlgorithm end

function CommonSolve.solve(prob::IntervalNonlinearProblem, alg::Brent, args...;
        maxiters = 1000, abstol = nothing, kwargs...)
    @assert !SciMLBase.isinplace(prob) "`Brent` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)
    ϵ = eps(convert(typeof(fl), 1))

    abstol = NonlinearSolveBase.get_tolerance(
        abstol, promote_type(eltype(left), eltype(right)))

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right)
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right)
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
                return SciMLBase.build_solution(prob, alg, left, fl;
                    retcode = ReturnCode.FloatingPointLimit, left, right)
            end
            cond = true
        else
            cond = false
        end

        fs = f(s)
        if abs((right - left) / 2) < abstol
            return SciMLBase.build_solution(
                prob, alg, s, fs; retcode = ReturnCode.Success, left, right)
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

    sol, i, left, right, fl, fr = Impl.bisection(
        left, right, fl, fr, f, abstol, maxiters - i, prob, alg)

    sol !== nothing && return sol

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
