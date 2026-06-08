using NonlinearSolve

using NonlinearSolve

# This test ensures the polyalgorithm fallback path works correctly.
# Previously, when all sub-algorithms failed to converge early, the code
# tried to access `cache.resid` which doesn't exist (caches have `fu`).
# This caused "type GeneralizedFirstOrderAlgorithmCache has no field resid".

# Create a problem that's hard to converge with very few iterations,
# forcing all polyalgorithm sub-algorithms to fail and trigger the fallback path.
function hard_problem(u, p)
    # Stiff problem that's hard to converge quickly
    return @. u^10 - p
end

prob = NonlinearLeastSquaresProblem(hard_problem, [100.0, 200.0], [1.0, 1.0])

# Use very few iterations to force all algorithms to fail early convergence
sol = solve(prob; maxiters = 3)

# The key test: this should not error with "no field resid"
# The solution should have a valid structure even if not converged
@test sol.u isa AbstractVector
@test sol.resid isa AbstractVector
@test sol.retcode == ReturnCode.MaxIters
