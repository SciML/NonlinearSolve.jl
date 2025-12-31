"""
    Alefeld()

An implementation of algorithm 4.2 from [Alefeld](https://dl.acm.org/doi/10.1145/210089.210111).

The paper brought up two new algorithms. Here choose to implement algorithm 4.2 rather than
algorithm 4.1 because, in certain sense, the second algorithm(4.2) is an optimal procedure.
"""
struct Alefeld <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Alefeld, args...;
        maxiters = 1000, abstol = nothing, kwargs...
)
    f = Base.Fix2(prob.f, prob.p)
    a, b = prob.tspan
    c = a - (b - a) / (f(b) - f(a)) * f(a)

    fc = f(c)
    if a == c || b == c
        return build_bracketing_solution(prob, alg, c, fc, a, b, ReturnCode.FloatingPointLimit)
    end

    if iszero(fc)
        return build_exact_solution(prob, alg, c, fc, ReturnCode.Success)
    end

    a, b, d = Impl.bracket(f, a, b, c)
    e = zero(a)   # Set e as 0 before iteration to avoid a non-value f(e)

    for i in 2:maxiters
        # The first bracketing block
        f₁, f₂, f₃, f₄ = f(a), f(b), f(d), f(e)
        if i == 2 || (f₁ == f₂ || f₁ == f₃ || f₁ == f₄ || f₂ == f₃ || f₂ == f₄ || f₃ == f₄)
            c = Impl.newton_quadratic(f, a, b, d, 2)
        else
            c = Impl.ipzero(f, a, b, d, e)
            if (c - a) * (c - b) ≥ 0
                c = Impl.newton_quadratic(f, a, b, d, 2)
            end
        end

        ē, fc = d, f(c)
        if a == c || b == c
            return build_bracketing_solution(prob, alg, c, fc, a, b, ReturnCode.FloatingPointLimit)
        end

        if iszero(fc)
            return build_exact_solution(prob, alg, c, fc, ReturnCode.Success)
        end

        ā, b̄, d̄ = Impl.bracket(f, a, b, c)

        # The second bracketing block
        f₁, f₂, f₃, f₄ = f(ā), f(b̄), f(d̄), f(ē)
        if f₁ == f₂ || f₁ == f₃ || f₁ == f₄ || f₂ == f₃ || f₂ == f₄ || f₃ == f₄
            c = Impl.newton_quadratic(f, ā, b̄, d̄, 3)
        else
            c = Impl.ipzero(f, ā, b̄, d̄, ē)
            if (c - ā) * (c - b̄) ≥ 0
                c = Impl.newton_quadratic(f, ā, b̄, d̄, 3)
            end
        end
        fc = f(c)

        if ā == c || b̄ == c
            return build_bracketing_solution(prob, alg, c, fc, ā, b̄, ReturnCode.FloatingPointLimit)
        end

        if iszero(fc)
            return build_exact_solution(prob, alg, c, fc, ReturnCode.Success)
        end

        ā, b̄, d̄ = Impl.bracket(f, ā, b̄, c)

        # The third bracketing block
        u = ifelse(abs(f(ā)) < abs(f(b̄)), ā, b̄)
        c = u - 2 * (b̄ - ā) / (f(b̄) - f(ā)) * f(u)
        if (abs(c - u)) > 0.5 * (b̄ - ā)
            c = 0.5 * (ā + b̄)
        end
        fc = f(c)

        if ā == c || b̄ == c
            return build_bracketing_solution(prob, alg, c, fc, ā, b̄, ReturnCode.FloatingPointLimit)
        end

        if iszero(fc)
            return build_exact_solution(prob, alg, c, fc, ReturnCode.Success)
        end

        ā, b̄, d = Impl.bracket(f, ā, b̄, c)

        # The last bracketing block
        if b̄ - ā < 0.5 * (b - a)
            a, b, e = ā, b̄, d̄
        else
            e = d
            c = 0.5 * (ā + b̄)
            fc = f(c)

            if ā == c || b̄ == c
                return build_bracketing_solution(prob, alg, c, fc, ā, b̄, ReturnCode.FloatingPointLimit)
            end
            if iszero(fc)
                return build_exact_solution(prob, alg, c, fc, ReturnCode.Success)
            end
            a, b, d = Impl.bracket(f, ā, b̄, c)
        end
    end

    # Reassign the value a, b, and c
    if b == c
        b = d
    elseif a == c
        a = d
    end
    fc = f(c)

    # Return solution when run out of max iteration
    return build_bracketing_solution(prob, alg, c, fc, a, b, ReturnCode.MaxIters)
end
