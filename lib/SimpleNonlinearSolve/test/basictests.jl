using SimpleNonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

# SimpleNewtonRaphson
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, SimpleNewtonRaphson()))
end

function ff(u, p)
    u .* u .- 2
end
const cu0 = @SVector[1.0, 1.0]
function sf(u, p)
    u * u - 2
end
const csu0 = 1.0

sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9

if VERSION >= v"1.7"
    @test (@ballocated benchmark_scalar(sf, csu0)) == 0
end

# Broyden
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, Broyden()))
end

sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9
if VERSION >= v"1.7"
    @test (@ballocated benchmark_scalar(sf, csu0)) == 0
end

# Klement
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, Klement()))
end

sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9
if VERSION >= v"1.7"
    @test (@ballocated benchmark_scalar(sf, csu0)) == 0
end

# SimpleTrustRegion
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, SimpleTrustRegion()))
end

sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

for alg in (SimpleNewtonRaphson(), Broyden(), Klement(), SimpleTrustRegion())
    g = function (p)
        probN = NonlinearProblem{false}(f, csu0, p)
        sol = solve(probN, alg, abstol = 1e-9)
        return sol.u[end]
    end

    for p in 1.1:0.1:100.0
        @test g(p) ≈ sqrt(p)
        @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
    end
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0
for alg in (SimpleNewtonRaphson(), Broyden(), Klement(), SimpleTrustRegion())
    g = function (p)
        probN = NonlinearProblem{false}(f, oftype(p, u0), p)
        sol = solve(probN, alg)
        return sol.u
    end

    for p in 1.1:0.1:100.0
        @test g(p) ≈ sqrt(p)
        @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
    end
end

tspan = (1.0, 20.0)
# Falsi
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
    sol = solve(probN, Falsi())
    return sol.left
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Ridder
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
    sol = solve(probN, Ridder())
    return sol.left
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

f, tspan = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
for alg in [Bisection(), Falsi(), Ridder()]
    global g, p
    g = function (p)
        probN = IntervalNonlinearProblem{false}(f, tspan, p)
        sol = solve(probN, alg)
        return [sol.left]
    end

    @test g(p) ≈ [sqrt(p[2] / p[1])]
    @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
end

for alg in (SimpleNewtonRaphson(), Broyden(), Klement(), SimpleTrustRegion())
    global g, p
    g = function (p)
        probN = NonlinearProblem{false}(f, 0.5, p)
        sol = solve(probN, alg)
        return [sol.u]
    end
    @test g(p) ≈ [sqrt(p[2] / p[1])]
    @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
end

# Error Checks
f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, SimpleNewtonRaphson()).u[end] ≈ sqrt(2.0)
@test solve(probN, SimpleNewtonRaphson(; autodiff = false)).u[end] ≈ sqrt(2.0)
@test solve(probN, SimpleTrustRegion()).u[end] ≈ sqrt(2.0)
@test solve(probN, SimpleTrustRegion(; autodiff = false)).u[end] ≈ sqrt(2.0)
@test solve(probN, Broyden()).u[end] ≈ sqrt(2.0)
@test solve(probN, Klement()).u[end] ≈ sqrt(2.0)

for u0 in [1.0, [1, 1.0]]
    local f, probN, sol
    f = (u, p) -> u .* u .- 2.0
    probN = NonlinearProblem(f, u0)
    sol = sqrt(2) * u0

    @test solve(probN, SimpleNewtonRaphson()).u ≈ sol
    @test solve(probN, SimpleNewtonRaphson()).u ≈ sol
    @test solve(probN, SimpleNewtonRaphson(; autodiff = false)).u ≈ sol

    @test solve(probN, SimpleTrustRegion()).u ≈ sol
    @test solve(probN, SimpleTrustRegion()).u ≈ sol
    @test solve(probN, SimpleTrustRegion(; autodiff = false)).u ≈ sol

    @test solve(probN, Broyden()).u ≈ sol

    @test solve(probN, Klement()).u ≈ sol
end

# Bisection Tests
f, tspan = (u, p) -> u .* u .- 2.0, (1.0, 2.0)
probB = IntervalNonlinearProblem(f, tspan)

# Falsi
sol = solve(probB, Falsi())
@test sol.left ≈ sqrt(2.0)

sol = solve(probB, Bisection())
@test sol.left ≈ sqrt(2.0)

# Ridder
sol = solve(probB, Ridder())
@test sol.left ≈ sqrt(2.0)
tspan = (sqrt(2.0), 10.0)
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Ridder())
@test sol.left ≈ sqrt(2.0)
tspan = (0.0, sqrt(2.0))
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Ridder())
@test sol.left ≈ sqrt(2.0)

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
probB = IntervalNonlinearProblem(f, (0.0, 4.0))

sol = solve(probB, Bisection(; exact_left = true))
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0

sol = solve(probB, Bisection(; exact_right = true))
@test f(sol.right, nothing) >= 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0

sol = solve(probB, Bisection(; exact_left = true, exact_right = true); immutable = false)
@test f(sol.left, nothing) < 0.0
@test f(nextfloat(sol.left), nothing) >= 0.0
@test f(sol.right, nothing) >= 0.0
@test f(prevfloat(sol.right), nothing) <= 0.0

# Test that `SimpleTrustRegion` passes a test that `SimpleNewtonRaphson` fails on.
u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
global g, f
f = (u, p) -> 0.010000000000000002 .+
              10.000000000000002 ./ (1 .+
               (0.21640425613334457 .+
                216.40425613334457 ./ (1 .+
                 (0.21640425613334457 .+
                  216.40425613334457 ./
                  (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^ 2.0) .-
              0.0011552453009332421u .- p
g = function (p)
    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, SimpleTrustRegion())
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

# Test kwars in `SimpleTrustRegion`
max_trust_radius = [10.0, 100.0, 1000.0]
initial_trust_radius = [10.0, 1.0, 0.1]
step_threshold = [0.0, 0.01, 0.25]
shrink_threshold = [0.25, 0.3, 0.5]
expand_threshold = [0.5, 0.8, 0.9]
shrink_factor = [0.1, 0.3, 0.5]
expand_factor = [1.5, 2.0, 3.0]
max_shrink_times = [10, 20, 30]

list_of_options = zip(max_trust_radius, initial_trust_radius, step_threshold,
                      shrink_threshold, expand_threshold, shrink_factor,
                      expand_factor, max_shrink_times)
for options in list_of_options
    local probN, sol, alg
    alg = SimpleTrustRegion(max_trust_radius = options[1],
                            initial_trust_radius = options[2],
                            step_threshold = options[3],
                            shrink_threshold = options[4],
                            expand_threshold = options[5],
                            shrink_factor = options[6],
                            expand_factor = options[7],
                            max_shrink_times = options[8])

    probN = NonlinearProblem(f, u0, p)
    sol = solve(probN, alg)
    @test all(abs.(f(u, p)) .< 1e-10)
end
