"""
Builder for exact bracketing problems solution.
"""
function build_exact_solution(prob, alg, u, resid, retcode)
    return SciMLBase.build_solution(
        prob, alg, u, resid; retcode, left = u, right = u
    )
end

"""
Builder for bracketing problems solution.
Ensure that left/right are in the same order as tspan for consistency.
"""
function build_bracketing_solution(prob, alg, u, resid, bound1, bound2, retcode)
    return if xor(bound1 < bound2, prob.tspan[1] < prob.tspan[2])
        SciMLBase.build_solution(
            prob, alg, u, resid; retcode, left = bound2, right = bound1
        )
    else
        SciMLBase.build_solution(
            prob, alg, u, resid; retcode, left = bound1, right = bound2
        )
    end
end

"""
    bracket_endpoints(tspan)

The two entries of `tspan`, promoted to one type and ordered. Any iterable with exactly two
entries works; a third entry is an error, raised before the residual is evaluated.
"""
function bracket_endpoints(tspan)
    next1 = iterate(tspan)
    next1 === nothing && throw(bracket_arity_error())
    next2 = iterate(tspan, next1[2])
    next2 === nothing && throw(bracket_arity_error())
    iterate(tspan, next2[2]) === nothing || throw(bracket_arity_error())
    return minmax(promote(next1[1], next2[1])...)
end

bracket_arity_error() = ArgumentError("an interval bracket has exactly two entries")
