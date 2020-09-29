using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(), immutable = true, tol = 1e-9)
    sol = solve!(solver)
end

function benchmark_mutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(), immutable = false, tol = 1e-9)
    sol = (reinit!(solver, probN); solve!(solver))
end

function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, NewtonRaphson()))
end

f, u0 = (u,p) -> u .* u .- 2, @SVector[1.0, 1.0]
sf, su0 = (u,p) -> u * u - 2, 1.0
sol = benchmark_immutable(f, u0)
@test sol.retcode === :Default
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(f, u0)
@test sol.retcode === :Default
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, su0)
@test sol.retcode === :Default
@test sol.u * sol.u - 2 < 1e-9

@test (@ballocated benchmark_immutable($f, $u0)) == 0
@test (@ballocated benchmark_mutable($f, $u0)) < 200
@test (@ballocated benchmark_scalar($sf, $su0)) == 0

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, NewtonRaphson(), immutable = true, tol = 1e-9)
    return sol.u[end]
end

for p in 1.0:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1/(2*sqrt(p))
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0

g = function (p)
    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, NewtonRaphson())
    return sol.u
end

@test_broken ForwardDiff.derivative(g, 1.0) ≈ 0.5

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1/(2*sqrt(p))
end
