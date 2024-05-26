"""
    Brent()

left non-allocating Brent method.
"""
struct Brent <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Brent, args...;
        maxiters = 1000, abstol = nothing, kwargs...)
    @assert !isinplace(prob) "`Brent` only supports OOP problems."
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)
    ϵ = eps(convert(typeof(fl), 1))

    abstol = __get_tolerance(
        nothing, abstol, promote_type(eltype(first(prob.tspan)), eltype(last(prob.tspan))))

    if iszero(fl)
        return build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right)
    end

    if iszero(fr)
        return build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right)
    end

    if abs(fl) < abs(fr)
        c = right
        right = left
        left = c
        tmp = fl
        fl = fr
        fr = tmp
    end

    c = left
    d = c
    i = 1
    cond = true
    if !iszero(fr)
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
                (s == left || s == right) && return SciMLBase.build_solution(
                    prob, alg, left, fl; retcode = ReturnCode.FloatingPointLimit,
                    left = left, right = right)
                cond = true
            else
                cond = false
            end
            fs = f(s)
            if abs((right - left) / 2) < abstol
                return SciMLBase.build_solution(
                    prob, alg, s, fs; retcode = ReturnCode.Success,
                    left = left, right = right)
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
                d = c
                c = right
                right = s
                fr = fs
            else
                left = s
                fl = fs
            end
            if abs(fl) < abs(fr)
                d = c
                c = right
                right = left
                left = c
                fc = fr
                fr = fl
                fl = fc
            end
            i += 1
        end
    end

    sol, i, left, right, fl, fr = __bisection(
        left, right, fl, fr, f; abstol, maxiters = maxiters - i, prob, alg)

    sol !== nothing && return sol

    return build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
