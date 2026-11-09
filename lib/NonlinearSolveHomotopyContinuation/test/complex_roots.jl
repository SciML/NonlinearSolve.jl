using NonlinearSolve
using NonlinearSolveHomotopyContinuation
using SciMLBase: NonlinearSolution

# Test complex roots for scalar polynomial
@testset "Complex roots - scalar" begin
    # Polynomial: u^2 + 1 = 0, roots should be ±i
    rhs = function (u, p)
        return u * u + 1
    end

    prob = NonlinearProblem(rhs, 1.0 + 0.0im)

    # Test with complex roots enabled
    alg_complex = HomotopyContinuationJL{true, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa EnsembleSolution
    @test sol_complex.converged
    @test length(sol_complex.u) == 2

    # Sort solutions by imaginary part
    solutions = [s.u for s in sol_complex.u]
    sort!(solutions; by = imag)

    @test solutions[1] ≈ -1im atol = 1.0e-10
    @test solutions[2] ≈ 1im atol = 1.0e-10

    # Test with complex roots disabled (should find no real solutions)
    alg_real = HomotopyContinuationJL{true, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !sol_real.converged
    @test length(sol_real.u) == 1
    @test sol_real.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure
end

# Test complex roots for vector polynomial
@testset "Complex roots - vector" begin
    # System: u[1]^2 + 1 = 0, u[2]^2 + 4 = 0
    # Roots should be [±i, ±2i]
    rhs = function (u, p)
        return [u[1]^2 + 1, u[2]^2 + 4]
    end

    prob = NonlinearProblem(rhs, [1.0 + 0.0im, 1.0 + 0.0im])

    # Test with complex roots enabled
    alg_complex = HomotopyContinuationJL{true, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa EnsembleSolution
    @test sol_complex.converged
    @test length(sol_complex.u) == 4

    # Verify all solutions are approximately correct
    for s in sol_complex.u
        u = s.u
        @test abs(u[1]^2 + 1) < 1.0e-10
        @test abs(u[2]^2 + 4) < 1.0e-10
    end

    # Test with complex roots disabled (should find no real solutions)
    alg_real = HomotopyContinuationJL{true, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !sol_real.converged
    @test length(sol_real.u) == 1
    @test sol_real.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure
end

# Test single root method with complex roots
@testset "Complex roots - single root" begin
    # Polynomial: u^2 + 1 = 0
    rhs = function (u, p)
        return u * u + 1
    end

    # H(x, t) = F(x) - t*F(u0) is singular where F'(u) = 0, i.e. at u = 0, which lies
    # on the path iff t = 1/F(u0) is real and in [0, 1]. A purely real guess gives
    # F(u0) = u0^2 + 1 > 1, so the path crosses the singularity at t = 0.5 and the
    # tracker fails; a guess with non-real F(u0) keeps the path regular.
    prob = NonlinearProblem(rhs, 0.5 + 0.9im)

    # Test with complex roots enabled
    alg_complex = HomotopyContinuationJL{false, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa NonlinearSolution
    @test SciMLBase.successful_retcode(sol_complex)
    @test abs(sol_complex.u^2 + 1) < 1.0e-10

    # Test with complex roots disabled
    alg_real = HomotopyContinuationJL{false, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !SciMLBase.successful_retcode(sol_real)
end
