using NonlinearSolve

using NonlinearSolve

# This test ensures the polyalgorithm cache-based solve! works correctly.
# Issue #779: PR #773 changed line 386 to use get_fu(cache) to extract residuals,
# but line 391 still called findmin_caches(cache.prob, fus) with the residual
# vectors. Since findmin_caches internally calls get_fu() expecting cache objects,
# this caused "MethodError: no method matching get_fu(::Vector{Float64})".
# The fix uses findmin_resids directly since fus already contains residual vectors.

function hard_problem(u, p)
    return @. u^10 - p
end

prob = NonlinearLeastSquaresProblem(hard_problem, [100.0, 200.0], [1.0, 1.0])

# Initialize cache and use solve! to exercise the cache-based code path
# The key test: this should not error with "no method matching get_fu(::Vector{Float64})"
cache = init(prob; maxiters = 3)
sol = solve!(cache)

@test sol.u isa AbstractVector
@test sol.resid isa AbstractVector
@test SciMLBase.successful_retcode(sol) || sol.retcode == ReturnCode.MaxIters
