using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, NewtonRaphson(), tol = 1e-9)
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
@test sol.retcode === NonlinearSolve.DEFAULT
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(f, u0)
@test sol.retcode === NonlinearSolve.DEFAULT
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, su0)
@test sol.retcode === NonlinearSolve.DEFAULT
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

# Error Checks

f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, NewtonRaphson()).u[end] ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(); immutable = false).u[end] ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(;autodiff=false)).u[end] ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(;autodiff=false); immutable = false).u[end] ≈ sqrt(2.0)

f, u0 = (u, p) -> u .* u .- 2.0, 1.0
probN = NonlinearProblem(f, u0)

@test solve(probN, NewtonRaphson()).u ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(); immutable = false).u ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(;autodiff=false)).u ≈ sqrt(2.0)
@test solve(probN, NewtonRaphson(;autodiff=false); immutable = false).u ≈ sqrt(2.0)


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
# Question: Do we need BracketingImmutableSolver? We have fast scalar overload and 
# Bracketing solvers work only for scalars.

solver = init(probB, Bisection(); immutable = false)
# @test solver isa NonlinearSolve.BracketingSolver
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

solver = init(probB, Bisection(;exact_left = true); immutable = false)
sol = solve!(solver)
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0

solver = init(probB, Bisection(;exact_right = true); immutable = false)
sol = solve!(solver)
@test f(sol.right, nothing) > 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0

solver = init(probB, Bisection(;exact_left = true, exact_right = true); immutable = false)
sol = solve!(solver)
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0
@test f(sol.right, nothing) > 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0
