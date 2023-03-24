#struct Alefeld <: AbstractSimpleNonlinearSolveAlgorithm end

# Define subrotine function bracket, check d to see whether the zero is found when using.
function _bracket(f::Function, a, b, c)
    fc = f(c)
    if fc == 0
        ā, b̄, d = a, b, c
    else
        fa, fb = f(a), f(b)
        if fa * fc < 0 
            ā, b̄, d = a, c, b
        elseif fb * fc < 0
            ā, b̄, d = c, b, a
        end
    end
    return ā, b̄, d 
end

# Define subrotine function 
#function _newton_quadratic()



# test 
function fk(x)
    return 2 * x
end

_bracket(fk, -2, 2, 0)