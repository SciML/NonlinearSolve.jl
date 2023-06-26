using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

using SciMLNLSolve

###-----Trust Region tests-----###

# some simple functions #
function f_oop(u, p)
    u .* u .- p
end

function f_iip(du, u, p)
    du .= u .* u .- p
end

function f_scalar(u, p)
    u * u - p
end

u0 = [1.0, 1.0]
csu0 = 1.0
p = [2.0, 2.0]
radius_update_scheme = RadiusUpdateSchemes.Simple
tol = 1e-9

function convergence_test_oop(f, u0, p, radius_update_scheme)
    prob = NonlinearProblem{false}(f, oftype(p, u0), p)
    cache = init(prob,
        TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9)
    sol = solve!(cache)
    return cache.internalnorm(cache.u_prev - cache.u), cache.iter, sol.retcode
end

residual, iterations, return_code = convergence_test_oop(f_oop, u0, p, radius_update_scheme)
@test return_code === ReturnCode.Success
@test residual ≈ tol
