using NonlinearSolve

using SciMLBase, NonlinearSolveBase

# Test out-of-place version
f(u, p) = u .- p
u0 = [5.0, 5.0]
p = [1.0, 2.0]
nf = NonlinearFunction(f; resid_prototype = zeros(2))
alg = LevenbergMarquardt()

@test !SciMLBase.allowsbounds(alg)

@testset "solution within bounds" begin
    prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [0.0, 0.0], ub = [10.0, 10.0])
    sol = solve(prob, alg)

    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
    @test all(sol.u .>= 0.0)
    @test all(sol.u .<= 10.0)
end

@testset "solution clamped to bounds" begin
    prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [3.0, 3.0], ub = [10.0, 10.0])
    sol = solve(prob, alg)
    @test all(sol.u .>= 3.0)
    @test all(sol.u .<= 10.0)

    # Test init + solve! path
    cache = init(prob, alg)
    sol = solve!(cache)
    @test all(sol.u .>= 3.0)
    @test all(sol.u .<= 10.0)
end

# Test in-place version — use FullSpecialize for bounds compatibility
f!(resid, u, p) = resid .= u .- p
u0 = [5.0, 5.0]
p = [1.0, 2.0]
nf = NonlinearFunction{true, SciMLBase.FullSpecialize}(f!)
lb = [0.0, 0.0]
ub = [10.0, 10.0]

prob = NonlinearLeastSquaresProblem(nf, u0, p; lb, ub)
sol = solve(prob, alg)
@test SciMLBase.successful_retcode(sol)
@test sol.u ≈ [1.0, 2.0] atol = 1.0e-6

# FullSpecialize preserves function identity
@test sol.prob.f.f === f!

# Default (AutoSpecialize) also works with bounds — bounds transform
# unwraps AutoSpecializeCallable before wrapping in BoundedWrapper
prob_as = NonlinearLeastSquaresProblem(NonlinearFunction(f!), u0, p; lb, ub)
sol_as = solve(prob_as, alg)
@test SciMLBase.successful_retcode(sol_as)
@test sol_as.u ≈ [1.0, 2.0] atol = 1.0e-6
@test sol.prob.lb == lb
@test sol.prob.ub == ub
@test sol.prob.u0 == prob.u0
