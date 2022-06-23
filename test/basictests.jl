using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(); tol = 1e-9)
    return sol = solve!(solver)
end

function benchmark_mutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(); tol = 1e-9)
    return sol = (reinit!(solver, probN); solve!(solver))
end

function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    return sol = (solve(probN, NewtonRaphson()))
end

function ff(u, p)
    return u .* u .- 2
end
const cu0 = @SVector[1.0, 1.0]
function sf(u, p)
    return u * u - 2
end
const csu0 = 1.0

sol = benchmark_immutable(ff, cu0)
@test sol.retcode === Symbol(NonlinearSolve.DEFAULT)
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(ff, cu0)
@test sol.retcode === Symbol(NonlinearSolve.DEFAULT)
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === Symbol(NonlinearSolve.DEFAULT)
@test sol.u * sol.u - 2 < 1e-9

@test (@ballocated benchmark_immutable(ff, cu0)) == 0
@test (@ballocated benchmark_mutable(ff, cu0)) < 200
@test (@ballocated benchmark_scalar(sf, csu0)) == 0

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, NewtonRaphson(); tol = 1e-9)
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
    sol = solve(probN, NewtonRaphson())
    return sol.u
end

@test ForwardDiff.derivative(g, 1.0) ≈ 0.5

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

u0 = (1.0, 20.0)
# Falsi
g = function (p)
    probN = NonlinearProblem{false}(f, typeof(p).(u0), p)
    sol = solve(probN, Falsi())
    return sol.left
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

f, u0 = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
for alg in [Bisection(), Falsi()]
    global g, p
    g = function (p)
        probN = NonlinearProblem{false}(f, u0, p)
        sol = solve(probN, Bisection())
        return [sol.left]
    end

    @test g(p) ≈ [sqrt(p[2] / p[1])]
    @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
end

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
@test solve(probN, NewtonRaphson(); immutable = false).u[end] ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(; autodiff = false)).u[end] ≈ sqrt(2.0)
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

# Bisection Tests
f, u0 = (u, p) -> u .* u .- 2.0, (1.0, 2.0)
probB = NonlinearProblem(f, u0)

# Falsi
solver = init(probB, Falsi())
sol = solve!(solver)
@test sol.left ≈ sqrt(2.0)

# this should call the fast scalar overload
@test solve(probB, Bisection()).left ≈ sqrt(2.0)

# these should call the iterator version
solver = init(probB, Bisection())
@test solver isa NonlinearSolve.BracketingImmutableSolver
@test solve!(solver).left ≈ sqrt(2.0)

# Garuntee Tests for Bisection
f = function (u, p)
    if u < 2.0
        return u - 2.0
    elseif u > 3.0
        return u - 3.0
    else
        return 0.0
    end
end
probB = NonlinearProblem(f, (0.0, 4.0))

solver = init(probB, Bisection(; exact_left = true))
sol = solve!(solver)
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0

solver = init(probB, Bisection(; exact_right = true))
sol = solve!(solver)
@test f(sol.right, nothing) > 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0

solver = init(probB, Bisection(; exact_left = true, exact_right = true); immutable = false)
sol = solve!(solver)
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0
@test f(sol.right, nothing) > 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0
