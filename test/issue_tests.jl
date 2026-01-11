@testitem "Correct Best Solution: #565" tags = [:core] begin
    using NonlinearSolve, StableRNGs

    x = collect(0:0.1:10)

    line_fct(x, p) = p[1] .+ p[2] .* x

    y_line = line_fct(x, [1, 3])
    y_line_n = line_fct(x, [1, 3]) + randn(StableRNG(0), length(x))

    res(β, (x, y)) = line_fct(x, β) .- y

    prob = NonlinearLeastSquaresProblem(res, [1, 3], p = (x, y_line_n))
    sol1 = solve(prob; maxiters = 1000)

    prob = NonlinearLeastSquaresProblem(res, [1, 5], p = (x, y_line_n))
    sol2 = solve(prob; maxiters = 1000)

    @test sol1.u ≈ sol2.u
end

@testitem "Polyalgorithm Fallback Path: CurveFit.jl#76" tags = [:core] begin
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
end
