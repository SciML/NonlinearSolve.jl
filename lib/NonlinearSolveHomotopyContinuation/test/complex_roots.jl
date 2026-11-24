using NonlinearSolve
using NonlinearSolveHomotopyContinuation
using SciMLBase: NonlinearSolution

@testset "Complex roots - scalar" begin
    # u^2 + 1 = 0 has the two roots ±i
    rhs = function (u, p)
        return u * u + 1
    end

    prob = NonlinearProblem(rhs, 1.0 + 0.0im)

    alg_complex = HomotopyContinuationJL{true, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa EnsembleSolution
    @test sol_complex.converged
    @test length(sol_complex.u) == 2

    solutions = [s.u for s in sol_complex.u]
    sort!(solutions; by = imag)

    @test solutions[1] ≈ -1im atol = 1.0e-10
    @test solutions[2] ≈ 1im atol = 1.0e-10

    alg_real = HomotopyContinuationJL{true, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !sol_real.converged
    @test length(sol_real.u) == 1
    @test sol_real.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure
end

@testset "Complex roots - vector" begin
    # u[1]^2 + 1 = 0 and u[2]^2 + 4 = 0 have the four roots [±i, ±2i]
    rhs = function (u, p)
        return [u[1]^2 + 1, u[2]^2 + 4]
    end

    prob = NonlinearProblem(rhs, [1.0 + 0.0im, 1.0 + 0.0im])

    alg_complex = HomotopyContinuationJL{true, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa EnsembleSolution
    @test sol_complex.converged
    @test length(sol_complex.u) == 4

    for s in sol_complex.u
        u = s.u
        @test abs(u[1]^2 + 1) < 1.0e-10
        @test abs(u[2]^2 + 4) < 1.0e-10
    end

    solutions = [s.u for s in sol_complex.u]
    sort!(solutions; by = u -> (imag(u[1]), imag(u[2])))
    @test solutions[1] ≈ [-1im, -2im] atol = 1.0e-10
    @test solutions[2] ≈ [-1im, 2im] atol = 1.0e-10
    @test solutions[3] ≈ [1im, -2im] atol = 1.0e-10
    @test solutions[4] ≈ [1im, 2im] atol = 1.0e-10

    alg_real = HomotopyContinuationJL{true, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !sol_real.converged
    @test length(sol_real.u) == 1
    @test sol_real.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure
end

@testset "Complex roots - single root" begin
    rhs = function (u, p)
        return u * u + 1
    end

    # H(x, t) = F(x) - t*F(u0) is singular at u = 0, which lies on the tracked
    # path at t = 1/(u0^2 + 1) whenever F(u0) is real. Every real start satisfies
    # that, so a start with non-real F(u0) is needed to keep the path regular.
    prob = NonlinearProblem(rhs, 0.5 + 0.9im)

    alg_complex = HomotopyContinuationJL{false, Val{true}}(; threading = false)
    sol_complex = solve(prob, alg_complex)

    @test sol_complex isa NonlinearSolution
    @test SciMLBase.successful_retcode(sol_complex)
    @test abs(sol_complex.u^2 + 1) < 1.0e-10

    alg_real = HomotopyContinuationJL{false, Val{false}}(; threading = false)
    sol_real = solve(prob, alg_real)

    @test !SciMLBase.successful_retcode(sol_real)
end

@testset "Complex roots - stored precision" begin
    alg = HomotopyContinuationJL{true, Val{true}}(; threading = false)

    # the root 1e-10im underflows Float16, so the stored element type must not be
    # taken from the starting value
    prob = NonlinearProblem((u, p) -> 1.0e10 * u - im, Float16(0.0))
    sol = solve(prob, alg)
    @test sol.converged
    @test length(sol.u) == 1
    @test sol.u[1].u ≈ 1.0e-10im atol = 1.0e-12
    @test abs(sol.u[1].resid) < 1.0e-6

    for prob in (
            NonlinearProblem((u, p) -> u^2 + 2, 0),
            NonlinearProblem(NonlinearFunction((u, p) -> u^2 + 2; jac = (u, p) -> 2u), 0),
        )
        sol = solve(prob, alg)
        @test sol.converged
        solutions = [s.u for s in sol.u]
        sort!(solutions; by = imag)
        @test solutions[1] ≈ -sqrt(2)im atol = 1.0e-10
        @test solutions[2] ≈ sqrt(2)im atol = 1.0e-10
    end
end

@testset "Complex roots - inplace residuals" begin
    # a real `resid_prototype` must not receive the complex residuals of the roots
    allroots_f = function (r, u, p)
        r[1] = u[1]^2 + 2
        return nothing
    end
    prob = NonlinearProblem(NonlinearFunction(allroots_f; resid_prototype = zeros(1)), [0.0])
    sol = solve(prob, HomotopyContinuationJL{true, Val{true}}(; threading = false))
    @test sol.converged
    @test length(sol.u) == 2
    solutions = [s.u for s in sol.u]
    sort!(solutions; by = u -> imag(u[1]))
    @test solutions[1] ≈ [-sqrt(2)im] atol = 1.0e-10
    @test solutions[2] ≈ [sqrt(2)im] atol = 1.0e-10
    @test all(s.resid isa AbstractVector{<:Complex} for s in sol.u)
    @test all(size(s.resid) == (1,) for s in sol.u)
    @test all(abs(s.resid[1]) < 1.0e-10 for s in sol.u)

    singleroot_f = function (r, u, p)
        r[1] = u[1]^2 + (1 + 4im) * u[1] + 2
        return nothing
    end
    # a start at exactly zero makes the HomotopyContinuation 2.17 tracker stop at t = 1
    prob = NonlinearProblem(
        NonlinearFunction(singleroot_f; resid_prototype = zeros(1)), [0.5]
    )
    sol = solve(prob, HomotopyContinuationJL{false, Val{true}}(; threading = false))
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u[1]^2 + (1 + 4im) * sol.u[1] + 2) < 1.0e-10
    @test sol.resid isa AbstractVector{<:Complex}
    @test size(sol.resid) == (1,)
    @test abs(sol.resid[1]) < 1.0e-10

    sol = solve(
        NonlinearProblem(NonlinearFunction(singleroot_f), [0.5]),
        HomotopyContinuationJL{false, Val{true}}(; threading = false)
    )
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u[1]^2 + (1 + 4im) * sol.u[1] + 2) < 1.0e-10
    @test sol.resid isa AbstractVector{<:Complex}
    @test abs(sol.resid[1]) < 1.0e-10
end

@testset "Complex roots - flag validation" begin
    for C in (false, Val(false), Val(true), Nothing)
        @test_throws ArgumentError HomotopyContinuationJL{true, C}(; threading = false)
    end
    @test HomotopyContinuationJL{true, Val{true}}(; threading = false) isa
        HomotopyContinuationJL{true, Val{true}}
    @test HomotopyContinuationJL{true, Val{false}}(; threading = false) isa
        HomotopyContinuationJL{true, Val{false}}
end
