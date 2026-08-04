using NonlinearSolve
using NonlinearSolveHomotopyContinuation
using SciMLBase: NonlinearSolution
using ADTypes
using LinearAlgebra: norm

allroots = TaylorHomotopyContinuationJL{true}(; threading = false)

function sorted_roots(sol)
    us = [s.u for s in sol.u]
    return sort(us; by = u -> u isa Number ? u : u[1])
end

@testset "scalar transcendental" begin
    # u = 2 sin(u): roots 0 and ±1.8954942670339809. The expansion point must
    # be central enough for the truncation to cover all roots.
    rhs = (u, p) -> u - p * sin(u)
    prob = NonlinearProblem(rhs, 0.5, 2.0)

    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 9, threading = false))
    @test sol isa EnsembleSolution
    @test sol.converged
    us = sorted_roots(sol)
    @test length(us) == 3
    if length(us) == 3
        @test us[1] ≈ -1.8954942670339809 atol = 1.0e-8
        @test us[2] ≈ 0.0 atol = 1.0e-8
        @test us[3] ≈ 1.8954942670339809 atol = 1.0e-8
    end
    @test all(s -> abs(s.resid) < 1.0e-8, sol.u)

    # single-root variant returns the root closest to the initial guess
    prob = NonlinearProblem(rhs, 1.7, 2.0)
    sol1 = solve(prob, TaylorHomotopyContinuationJL(; degree = 9, threading = false))
    @test sol1 isa NonlinearSolution
    @test SciMLBase.successful_retcode(sol1)
    @test sol1.u ≈ 1.8954942670339809 atol = 1.0e-8
end

@testset "polynomial system is solved exactly" begin
    # Himmelblau critical points: gradient system, degree 3, 9 real roots
    rhs = function (u, p)
        x, y = u
        return [
            4x * (x^2 + y - p[1]) + 2 * (x + y^2 - p[2]),
            2 * (x^2 + y - p[1]) + 4y * (x + y^2 - p[2]),
        ]
    end
    prob = NonlinearProblem(rhs, zeros(2), [11.0, 7.0])
    sol = solve(prob, allroots)
    @test sol.converged
    @test length(sol.u) == 9
    @test all(s -> norm(s.resid) < 1.0e-8, sol.u)
    # the four Himmelblau minima are among the roots
    for minimum_u in (
            [3.0, 2.0], [-2.805118, 3.131313], [-3.77931, -3.283186],
            [3.584428, -1.848127],
        )
        @test any(s -> isapprox(s.u, minimum_u; atol = 1.0e-5), sol.u)
    end
end

@testset "transcendental system, out of place" begin
    # from the HomotopyContinuation.jl docs; roots on the circle
    # x^2 + y^2 = log(3) with x + y = sin(3(x+y))
    rhs = function (u, p)
        x, y = u
        return [exp(x^2 + y^2) - p, x + y - sin(3 * (x + y))]
    end
    prob = NonlinearProblem(rhs, [0.5, -0.5], 3.0)
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 6, threading = false))
    @test sol.converged
    @test length(sol.u) >= 2
    @test all(s -> norm(s.resid) < 1.0e-8, sol.u)
end

@testset "transcendental system, in place" begin
    # 1D Bratu with two solution branches
    lam = 3.0
    n = 4
    rhs = function (du, u, p)
        h = 1 / (length(u) + 1)
        for i in eachindex(u)
            um = i == 1 ? zero(eltype(u)) : u[i - 1]
            up = i == length(u) ? zero(eltype(u)) : u[i + 1]
            du[i] = (um - 2u[i] + up) / h^2 + p * exp(u[i])
        end
        return nothing
    end
    prob = NonlinearProblem(rhs, fill(0.5, n), lam)
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 4, threading = false))
    @test sol.converged
    @test length(sol.u) == 2
    @test all(s -> norm(s.resid) < 1.0e-8, sol.u)

    # single root: returns one converged branch
    sol1 = solve(prob, TaylorHomotopyContinuationJL(; degree = 4, threading = false))
    @test sol1 isa NonlinearSolution
    @test SciMLBase.successful_retcode(sol1)
    @test norm(sol1.resid) < 1.0e-8
end

@testset "explicit jacobian is used" begin
    rhs = (u, p) -> [u[1]^2 - p, u[1] + u[2]]
    jac = (u, p) -> [2u[1] 0.0; 1.0 1.0]
    fn = NonlinearFunction(rhs; jac)
    prob = NonlinearProblem(fn, [1.0, -1.0], 4.0)
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 2, threading = false))
    @test sol.converged
    us = sorted_roots(sol)
    @test length(us) == 2
    @test us[1] ≈ [-2.0, 2.0] atol = 1.0e-8
    @test us[2] ≈ [2.0, -2.0] atol = 1.0e-8
end

@testset "no real roots" begin
    rhs = (u, p) -> u^2 + p
    prob = NonlinearProblem(rhs, 1.0, 4.0)
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 2, threading = false))
    @test sol isa EnsembleSolution
    @test !sol.converged
    @test sol.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure

    sol1 = solve(prob, TaylorHomotopyContinuationJL(; degree = 2, threading = false))
    @test sol1.retcode == SciMLBase.ReturnCode.ConvergenceFailure
end

@testset "singular root" begin
    # Powell singular: multiplicity-4 root at the origin; HC classifies the
    # surrogate root as "excess" and Newton only converges linearly
    rhs = function (u, p)
        return [
            u[1] + 10u[2],
            sqrt(5) * (u[3] - u[4]),
            (u[2] - 2u[3])^2,
            sqrt(10) * (u[1] - u[4])^2,
        ]
    end
    prob = NonlinearProblem(rhs, [3.0, -1.0, 0.0, 1.0], nothing)
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 2, threading = false))
    @test sol.converged
    @test any(s -> norm(s.u) < 1.0e-4, sol.u)
end

@testset "unpolynomialize" begin
    # roots of x^2 - 5x + 6 in transformed coordinates y = exp(x)
    rhs = (u, p) -> u^2 - p[1] * u + p[2]
    f = HomotopyNonlinearFunction(
        NonlinearFunction(rhs);
        polynomialize = (u, p) -> log(u),
        unpolynomialize = (u, p) -> [exp(u)]
    )
    prob = NonlinearProblem(f, exp(1.9), [5.0, 6.0])
    sol = solve(prob, TaylorHomotopyContinuationJL{true}(; degree = 2, threading = false))
    @test sol.converged
    us = sorted_roots(sol)
    @test length(us) == 2
    @test us[1] ≈ exp(2.0) atol = 1.0e-6
    @test us[2] ≈ exp(3.0) atol = 1.0e-6
end
