"""
Builder for exact bracketing problems solution.
"""
function build_exact_solution(prob, alg, u, resid, retcode)
    SciMLBase.build_solution(
        prob, alg, u, resid; retcode = retcode, left = u, right = u
    )
end

"""
Builder for bracketing problems solution.
Ensure that left/right are in the same order as tspan for consistency.
"""
function build_bracketing_solution(prob, alg, u, resid, bound1, bound2, retcode)
    if xor(bound1 < bound2, prob.tspan[1] < prob.tspan[2])
        SciMLBase.build_solution(
            prob, alg, u, resid; retcode = retcode, left = bound2, right = bound1
        )
    else
        SciMLBase.build_solution(
            prob, alg, u, resid; retcode = retcode, left = bound1, right = bound2
        )
    end
end