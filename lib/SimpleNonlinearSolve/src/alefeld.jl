"""
`Alefeld()` 

An implementation of algorithm 4.2 from [Alefeld](https://dl.acm.org/doi/10.1145/210089.210111).

The paper brought up two new algorithms. Here choose to implement algorithm 4.2 rather than 
algorithm 4.1 because, in certain sense, the second algorithm(4.2) is an optimal procedure.
"""
struct Alefeld <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem,
                            alg::Alefeld, args...; abstol = nothing,
                            reltol = nothing,
                            maxiters = 1000, kwargs...)
                            
    f = Base.Fix2(prob.f, prob.p)
    a, b = prob.tspan
    c = a - (b - a) / (f(b) - f(a)) * f(a)
    
    fc = f(c)
    if iszero(fc)
        return SciMLBase.build_solution(prob, alg, c, fc;
                                        retcode = ReturnCode.Success, 
                                        left = a,
                                        right = b)
    end
    a, b, d = _bracket(f, a, b, c)
    e = zero(a)   # Set e as 0 before iteration to avoid a non-value f(e)

    # Begin of algorithm iteration
    for i in 2:maxiters
        # The first bracketing block
        f₁, f₂, f₃, f₄ = f(a), f(b), f(d), f(e)
        if i == 2 || (f₁ == f₂ || f₁ == f₃ || f₁ == f₄ || f₂ == f₃ || f₂ == f₄ || f₃ == f₄)
            c = _newton_quadratic(f, a, b, d, 2)
        else 
            c = _ipzero(f, a, b, d, e)
            if (c - a) * (c - b) ≥ 0
                c = _newton_quadratic(f, a, b, d, 2)
            end
        end 
        ē, fc = d, f(c)
        (a == c || b == c) && 
            return SciMLBase.build_solution(prob, alg, c, fc;
                                            retcode = ReturnCode.FloatingPointLimit,
                                            left = a, 
                                            right = b)   
        iszero(fc) &&
            return SciMLBase.build_solution(prob, alg, c, fc;
                                        retcode = ReturnCode.Success, 
                                        left = a,
                                        right = b)
        ā, b̄, d̄ = _bracket(f, a, b, c) 

        # The second bracketing block
        f₁, f₂, f₃, f₄ = f(ā), f(b̄), f(d̄), f(ē)
        if f₁ == f₂ || f₁ == f₃ || f₁ == f₄ || f₂ == f₃ || f₂ == f₄ || f₃ == f₄
            c = _newton_quadratic(f, ā, b̄, d̄, 3)
        else 
            c = _ipzero(f, ā, b̄, d̄, ē)
            if (c - ā) * (c - b̄) ≥ 0
                c = _newton_quadratic(f, ā, b̄, d̄, 3)
            end
        end
        fc = f(c)
        (ā == c || b̄ == c) && 
            return SciMLBase.build_solution(prob, alg, c, fc;
                                            retcode = ReturnCode.FloatingPointLimit,
                                            left = ā, 
                                            right = b̄)
        iszero(fc) &&
            return SciMLBase.build_solution(prob, alg, c, fc;
                                        retcode = ReturnCode.Success, 
                                        left = ā,
                                        right = b̄)
        ā, b̄, d̄ = _bracket(f, ā, b̄, c) 

        # The third bracketing block
        if abs(f(ā)) < abs(f(b̄))
            u = ā
        else
            u = b̄
        end
        c = u - 2 * (b̄ - ā) / (f(b̄) - f(ā)) * f(u)
        if (abs(c - u)) > 0.5 * (b̄ - ā)
            c = 0.5 * (ā + b̄)
        end
        fc = f(c)
        (ā == c || b̄ == c) && 
            return SciMLBase.build_solution(prob, alg, c, fc;
                                            retcode = ReturnCode.FloatingPointLimit,
                                            left = ā, 
                                            right = b̄)
        iszero(fc) &&
            return SciMLBase.build_solution(prob, alg, c, fc;
                                        retcode = ReturnCode.Success, 
                                        left = ā, 
                                        right = b̄)
        ā, b̄, d = _bracket(f, ā, b̄, c) 

        # The last bracketing block
        if b̄ - ā < 0.5 * (b - a)
            a, b, e = ā, b̄, d̄
        else
            e = d
            c = 0.5 * (ā + b̄)
            fc = f(c)
            (ā == c || b̄ == c) && 
                return SciMLBase.build_solution(prob, alg, c, fc;
                                                retcode = ReturnCode.FloatingPointLimit,
                                                left = ā, 
                                                right = b̄)
            iszero(fc) &&
                return SciMLBase.build_solution(prob, alg, c, fc;
                                                retcode = ReturnCode.Success, 
                                                left = ā,
                                                right = b̄)
            a, b, d = _bracket(f, ā, b̄, c)
        end
    end

    # Reassign the value a, b, and c
    if b == c
        b = d
    elseif a == c
        a = d
    end
    fc = f(c)

    # Reuturn solution when run out of max interation
    return SciMLBase.build_solution(prob, alg, c, fc; retcode = ReturnCode.MaxIters,
                                    left = a, right = b)
end

# Define subrotine function bracket, check fc before bracket to return solution
function _bracket(f::F, a, b, c) where F
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

# Define subrotine function newton quadratic, return the approximation of zero
function _newton_quadratic(f::F, a, b, d, k) where F
    A = ((f(d) - f(b)) / (d - b) - (f(b) - f(a)) / (b - a)) / (d - a) 
    B = (f(b) - f(a)) / (b - a)

    if iszero(A)
        return a - (1 / B) * f(a)
    elseif A * f(a) > 0
        rᵢ₋₁ = a 
    else 
        rᵢ₋₁ = b
    end 

    for i in 1:k
        rᵢ = rᵢ₋₁ - (f(a) + B * (rᵢ₋₁ - a) + A * (rᵢ₋₁ - a) * (rᵢ₋₁ - b)) / (B + A * (2 * rᵢ₋₁ - a - b))
        rᵢ₋₁ = rᵢ
    end

    return rᵢ₋₁
end

# Define subrotine function ipzero, also return the approximation of zero
function _ipzero(f::F, a, b, c, d) where F
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