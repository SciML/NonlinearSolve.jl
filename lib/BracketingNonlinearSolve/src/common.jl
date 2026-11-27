module Impl

prevfloat_tdir(x, x0, x1) = ifelse(x1 > x0, prevfloat(x), nextfloat(x))
nextfloat_tdir(x, x0, x1) = ifelse(x1 > x0, nextfloat(x), prevfloat(x))
max_tdir(a, b, x0, x1) = ifelse(x1 > x0, max(a, b), min(a, b))

# `eps(x)` in the units of `x`: DynamicQuantities defines no `eps`, but every number type
# used here has `nextfloat` and `prevfloat`
ulp(x::AbstractFloat) = eps(x)
function ulp(x)
    y = abs(x)
    z = nextfloat(y)
    return isfinite(z) ? z - y : y - prevfloat(y)
end

# `abstol`, a tolerance in the units of x, as a tolerance on the residual `fx`. Values that
# compare directly (plain numbers of any precision, or Unitful quantities of one dimension
# such as m and km) are returned unchanged, so the tolerance keeps its precision and its
# physical size: it is never rounded to the type of `fx` (e.g. 1e5 overflows Float16, and
# 1e-8 is not an Int). Otherwise (e.g. x in m and f in N) the value of `abstol` is given the
# units of `fx`. DynamicQuantities types do not carry units, so its methods are in
# `BracketingNonlinearSolveDynamicQuantitiesExt`.
function residual_tolerance(abstol, fx)
    afx = abs(fx)
    isconcretetype(promote_type(typeof(abstol), typeof(afx))) && return abstol
    return abstol * (oneunit(afx) / oneunit(abstol))
end

# Sign tests against `zero(x)`, so that `x` may carry units
isnegative(x) = x < zero(x)
ispositive(x) = x > zero(x)
isnonnegative(x) = x ≥ zero(x)

function bracket(f::F, a, b, c) where {F}
    if iszero(f(c))
        ā, b̄, d = a, b, c
    else
        if isnegative(f(a) * f(c))
            ā, b̄, d = a, c, b
        elseif isnegative(f(b) * f(c))
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
    elseif ispositive(A * f(a))
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
