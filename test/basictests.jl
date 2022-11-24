using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_mutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, NewtonRaphson()))
end

function ff(u, p)
    u .* u .- 2
end
const cu0 = @SVector[1.0, 1.0]
function sf(u, p)
    u * u - 2
end
const csu0 = 1.0

sol = benchmark_immutable(ff, cu0)
@test sol.retcode === ReturnCode.Default
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(ff, cu0)
@test sol.retcode === ReturnCode.Default
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Default
@test sol.u * sol.u - 2 < 1e-9

@test (@ballocated benchmark_immutable(ff, cu0)) < 200
@test (@ballocated benchmark_mutable(ff, cu0)) < 200
@test (@ballocated benchmark_scalar(sf, csu0)) == 0

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, NewtonRaphson(), abstol = 1e-9)
    return sol.u[end]
end

for p in 1.0:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0

# NewtonRaphson
g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, NewtonRaphson(), abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 1.0) ≈ 0.5

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

f = (u, p) -> p[1] * u * u - p[2]
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
gnewton = function (p)
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, NewtonRaphson())
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# Error Checks

f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, NewtonRaphson()).u[end] ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(; autodiff = false)).u[end] ≈ sqrt(2.0)

for u0 in [1.0, [1, 1.0]]
    local f, probN, sol
    f = (u, p) -> u .* u .- 2.0
    probN = NonlinearProblem(f, u0)
    sol = sqrt(2) * u0

    @test solve(probN, NewtonRaphson()).u ≈ sol
    @test solve(probN, NewtonRaphson()).u ≈ sol
    @test solve(probN, NewtonRaphson(; autodiff = false)).u ≈ sol
end
