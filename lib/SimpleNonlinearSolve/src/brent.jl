"""
`Brent()`

A non-allocating Brent method

"""
struct Brent <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Brent, args...;
    maxiters = 1000, abstol = min(eps(prob.tspan[1]), eps(prob.tspan[2])),
    kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    a, b = prob.tspan
    fa, fb = f(a), f(b)
    ϵ = eps(convert(typeof(fa), 1.0))

    if iszero(fa)
        return SciMLBase.build_solution(prob, alg, a, fa;
            retcode = ReturnCode.ExactSolutionLeft, left = a,
            right = b)
    elseif iszero(fb)
        return SciMLBase.build_solution(prob, alg, b, fb;
            retcode = ReturnCode.ExactSolutionRight, left = a,
            right = b)
    end
    if abs(fa) < abs(fb)
        c = b
        b = a
        a = c
        tmp = fa
        fa = fb
        fb = tmp
    end

    c = a
    d = c
    i = 1
    cond = true
    if !iszero(fb)
        while i < maxiters
            fc = f(c)
            if fa != fc && fb != fc
                # Inverse quadratic interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                    b * fa * fc / ((fb - fa) * (fb - fc)) +
                    c * fa * fb / ((fc - fa) * (fc - fb))
            else
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            end
            if (s < min((3 * a + b) / 4, b) || s > max((3 * a + b) / 4, b)) ||
               (cond && abs(s - b) ≥ abs(b - c) / 2) ||
               (!cond && abs(s - b) ≥ abs(c - d) / 2) ||
               (cond && abs(b - c) ≤ ϵ) ||
               (!cond && abs(c - d) ≤ ϵ)
                # Bisection method
                s = (a + b) / 2
                (s == a || s == b) &&
                    return SciMLBase.build_solution(prob, alg, a, fa;
                        retcode = ReturnCode.FloatingPointLimit,
                        left = a, right = b)
                cond = true
            else
                cond = false
            end
            fs = f(s)
            if abs((b - a) / 2) < abstol
                return SciMLBase.build_solution(prob, alg, s, fs;
                    retcode = ReturnCode.Success,
                    left = a, right = b)
            end
            if iszero(fs)
                if b < a
                    a = b
                    fa = fb
                end
                b = s
                fb = fs
                break
            end
            if fa * fs < 0
                d = c
                c = b
                b = s
                fb = fs
            else
                a = s
                fa = fs
            end
            if abs(fa) < abs(fb)
                d = c
                c = b
                b = a
                a = c
                fc = fb
                fb = fa
                fa = fc
            end
            i += 1
        end
    end

    while i < maxiters
        c = (a + b) / 2
        if (c == a || c == b)
            return SciMLBase.build_solution(prob, alg, a, fa;
                retcode = ReturnCode.FloatingPointLimit,
                left = a, right = b)
        end
        fc = f(c)
        if abs((b - a) / 2) < abstol
            return SciMLBase.build_solution(prob, alg, c, fc;
                retcode = ReturnCode.Success,
                left = a, right = b)
        end
        if iszero(fc)
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
        i += 1
    end

    return SciMLBase.build_solution(prob, alg, a, fa; retcode = ReturnCode.MaxIters,
        left = a, right = b)
end
