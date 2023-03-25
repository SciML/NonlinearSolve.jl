#struct Alefeld <: AbstractSimpleNonlinearSolveAlgorithm end

# Define subrotine function bracket, check d to see whether the zero is found.
function _bracket(f::Function, a, b, c)
    if f(c) == 0
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

# Define subrotine function newton quadratic, return the approximation of unique zero.
function _newton_quadratic(f::Function, a, b, d, k)
    A = ((f(b) - f(d)) / (b - d) - (f(a) - f(b)) / (a - b)) / (d - a) 
    B = (f(b) - f(a)) / (b - a)
    if A == 0
        return a - (1 / B) * f(a)
    elseif A * f(a) > 0
        rᵢ₋₁ = a 
    else 
        rᵢ₋₁ = b
    end 
    for i in 1:k
        rᵢ = rᵢ₋₁ - B * rᵢ₋₁ / (B + A * (2 * rᵢ₋₁ - a - b))
        rᵢ₋₁ = rᵢ
    end
    return rᵢ₋₁
end


# test 
function fk(x)
    return x^3
end

_newton_quadratic(fk, -2, 4, 100, 2)

