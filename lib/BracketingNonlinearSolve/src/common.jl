module Impl

using SciMLBase: SciMLBase, ReturnCode

function bisection(left, right, fl, fr, f::F, abstol, maxiters, prob, alg) where {F}
    i = 1
    sol = nothing
    while i ≤ maxiters
        mid = (left + right) / 2

        if mid == left || mid == right
            sol = SciMLBase.build_solution(
                prob, alg, left, fl; left, right, retcode = ReturnCode.FloatingPointLimit
            )
            break
        end

        fm = f(mid)
        if abs((right - left) / 2) < abstol
            sol = SciMLBase.build_solution(
                prob, alg, mid, fm; left, right, retcode = ReturnCode.Success
            )
            break
        end

        if iszero(fm)
            right = mid
            fr = fm
        else
            left = mid
            fl = fm
        end

        i += 1
    end

    return sol, i, left, right, fl, fr
end

prevfloat_tdir(x, x0, x1) = ifelse(x1 > x0, prevfloat(x), nextfloat(x))
nextfloat_tdir(x, x0, x1) = ifelse(x1 > x0, nextfloat(x), prevfloat(x))
max_tdir(a, b, x0, x1) = ifelse(x1 > x0, max(a, b), min(a, b))

function bracket(f::F, a, b, c) where {F}
    if iszero(f(c))
        ā, b̄, d = a, b, c
    else
        if f(a) * f(c) < 0
            ā, b̄, d = a, c, b
        elseif f(b) * f(c) < 0
            ā, b̄, d = c, b, a
        end
    end
    return ā, b̄, d
end

function newton_quadratic(f::F, a, b, d, k) where {F}
    A = ((f(d) - f(b)) / (d - b) - (f(b) - f(a)) / (b - a)) / (d - a)
    B = (f(b) - f(a)) / (b - a)

    if iszero(A)
        return a - (1 / B) * f(a)
    elseif A * f(a) > 0
        rᵢ₋₁ = a
    else
        rᵢ₋₁ = b
    end

    for _ in 1:k
        rᵢ = rᵢ₋₁ -
             (f(a) + B * (rᵢ₋₁ - a) + A * (rᵢ₋₁ - a) * (rᵢ₋₁ - b)) /
             (B + A * (2 * rᵢ₋₁ - a - b))
        rᵢ₋₁ = rᵢ
    end

    return rᵢ₋₁
end

function ipzero(f::F, a, b, c, d) where {F}
    Q₁₁ = (c - d) * f(c) / (f(d) - f(c))
    Q₂₁ = (b - c) * f(b) / (f(c) - f(b))
    Q₃₁ = (a - b) * f(a) / (f(b) - f(a))
    D₂₁ = (b - c) * f(c) / (f(c) - f(b))
    D₃₁ = (a - b) * f(b) / (f(b) - f(a))
    Q₂₂ = (D₂₁ - Q₁₁) * f(b) / (f(d) - f(b))
    Q₃₂ = (D₃₁ - Q₂₁) * f(a) / (f(c) - f(a))
    D₃₂ = (D₃₁ - Q₂₁) * f(c) / (f(c) - f(a))
    Q₃₃ = (D₃₂ - Q₂₂) * f(a) / (f(d) - f(a))

    return a + Q₃₁ + Q₃₂ + Q₃₃
end

end
