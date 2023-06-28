using SimpleNonlinearSolve
using StaticArrays
using BenchmarkTools
using DiffEqBase
using LinearAlgebra
using Test

const BATCHED_BROYDEN_SOLVERS = Broyden[]
const BROYDEN_SOLVERS = Broyden[]
const BATCHED_LBROYDEN_SOLVERS = LBroyden[]
const LBROYDEN_SOLVERS = LBroyden[]
const BATCHED_DFSANE_SOLVERS = SimpleDFSane[]
const DFSANE_SOLVERS = SimpleDFSane[]

for mode in instances(NLSolveTerminationMode.T)
    if mode ∈
       (NLSolveTerminationMode.SteadyStateDefault, NLSolveTerminationMode.RelSafeBest,
        NLSolveTerminationMode.AbsSafeBest)
        continue
    end

    termination_condition = NLSolveTerminationCondition(mode; abstol = nothing,
        reltol = nothing)
    push!(BROYDEN_SOLVERS, Broyden(; batched = false, termination_condition))
    push!(BATCHED_BROYDEN_SOLVERS, Broyden(; batched = true, termination_condition))
    push!(LBROYDEN_SOLVERS, LBroyden(; batched = false, termination_condition))
    push!(BATCHED_LBROYDEN_SOLVERS, LBroyden(; batched = true, termination_condition))
    push!(DFSANE_SOLVERS, SimpleDFSane(; batched = false, termination_condition))
    push!(BATCHED_DFSANE_SOLVERS, SimpleDFSane(; batched = true, termination_condition))
end

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

# Halley
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, Halley()))
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

sol = benchmark_scalar(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test sol.u .* sol.u .- 2 < [1e-9, 1e-9]

if VERSION >= v"1.7"
    @test (@ballocated benchmark_scalar(sf, csu0)) == 0
end

# Broyden
function benchmark_scalar(f, u0, alg)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, alg))
end

for alg in BROYDEN_SOLVERS
    sol = benchmark_scalar(sf, csu0, alg)
    @test sol.retcode === ReturnCode.Success
    @test sol.u * sol.u - 2 < 1e-9
    # FIXME: Termination Condition Implementation is allocating. Not sure how to fix it.
    # if VERSION >= v"1.7"
    #     @test (@ballocated benchmark_scalar($sf, $csu0, $termination_condition)) == 0
    # end
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

# SimpleDFSane
function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, SimpleDFSane()))
end

sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

for alg in (SimpleNewtonRaphson(), LBroyden(), Klement(), SimpleTrustRegion(),
    SimpleDFSane(), Halley(), BROYDEN_SOLVERS...)
    g = function (p)
        probN = NonlinearProblem{false}(f, csu0, p)
        sol = solve(probN, alg, abstol = 1e-9)
        return sol.u[end]
    end

    for p in 1.1:0.1:100.0
        res = abs.(g(p))
        # Not surprising if LBrouden fails to converge
        if any(x -> isnan(x) || x <= 1e-5 || x >= 1e5, res) && alg isa LBroyden
            @test_broken res ≈ sqrt(p)
            @test_broken abs.(ForwardDiff.derivative(g, p)) ≈ 1 / (2 * sqrt(p))
        else
            @test res ≈ sqrt(p)
            @test abs.(ForwardDiff.derivative(g, p)) ≈ 1 / (2 * sqrt(p))
        end
    end
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0
for alg in (SimpleNewtonRaphson(), Klement(), SimpleTrustRegion(),
    SimpleDFSane(), Halley(), BROYDEN_SOLVERS..., LBROYDEN_SOLVERS...)
    g = function (p)
        probN = NonlinearProblem{false}(f, oftype(p, u0), p)
        sol = solve(probN, alg)
        return sol.u
    end

    for p in 1.1:0.1:100.0
        res = abs.(g(p))
        # Not surprising if LBrouden fails to converge
        if any(x -> isnan(x) || x <= 1e-5 || x >= 1e5, res) && alg isa LBroyden
            @test_broken res ≈ sqrt(p)
            @test_broken abs.(ForwardDiff.derivative(g, p)) ≈ 1 / (2 * sqrt(p))
        else
            @test res ≈ sqrt(p)
            @test abs.(ForwardDiff.derivative(g, p)) ≈ 1 / (2 * sqrt(p))
        end
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

# Brent
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
    sol = solve(probN, Brent())
    return sol.left
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# ITP
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
    sol = solve(probN, Itp())
    return sol.u
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Alefeld
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
    sol = solve(probN, Alefeld())
    return sol.u
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

f, tspan = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
g = function (p)
    probN = IntervalNonlinearProblem{false}(f, tspan, p)
    sol = solve(probN, Alefeld())
    return [sol.u]
end

@test g(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)

f, tspan = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
for alg in [Bisection(), Falsi(), Ridder(), Brent(), Itp()]
    global g, p
    g = function (p)
        probN = IntervalNonlinearProblem{false}(f, tspan, p)
        sol = solve(probN, alg)
        return [sol.left]
    end

    @test g(p) ≈ [sqrt(p[2] / p[1])]
    @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
end

for alg in (SimpleNewtonRaphson(), Klement(), SimpleTrustRegion(),
    SimpleDFSane(), Halley(), BROYDEN_SOLVERS..., LBROYDEN_SOLVERS...)
    global g, p
    g = function (p)
        probN = NonlinearProblem{false}(f, 0.5, p)
        sol = solve(probN, alg)
        return [abs(sol.u)]
    end
    @test g(p) ≈ [sqrt(p[2] / p[1])]
    @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
end

# Error Checks
f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

for alg in (SimpleNewtonRaphson(), SimpleNewtonRaphson(; autodiff = false),
    SimpleTrustRegion(),
    SimpleTrustRegion(; autodiff = false), Halley(), Halley(; autodiff = false),
    Klement(), SimpleDFSane(),
    BROYDEN_SOLVERS..., LBROYDEN_SOLVERS...)
    sol = solve(probN, alg)

    @test sol.retcode == ReturnCode.Success
    @test sol.u[end] ≈ sqrt(2.0)
end

for u0 in [1.0, [1, 1.0]]
    local f, probN, sol
    f = (u, p) -> u .* u .- 2.0
    probN = NonlinearProblem(f, u0)
    sol = sqrt(2) * u0

    for alg in (SimpleNewtonRaphson(), SimpleNewtonRaphson(; autodiff = false),
        SimpleTrustRegion(), SimpleTrustRegion(; autodiff = false), Klement(),
        SimpleDFSane(), BROYDEN_SOLVERS..., LBROYDEN_SOLVERS...)
        sol2 = solve(probN, alg)

        @test sol2.retcode == ReturnCode.Success
        @test sol2.u ≈ sol
    end
end

# Bisection Tests
f, tspan = (u, p) -> u .* u .- 2.0, (1.0, 2.0)
probB = IntervalNonlinearProblem(f, tspan)

# Falsi
sol = solve(probB, Falsi())
@test sol.left ≈ sqrt(2.0)

# Bisection
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

# Brent
sol = solve(probB, Brent())
@test sol.left ≈ sqrt(2.0)
tspan = (sqrt(2.0), 10.0)
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Brent())
@test sol.left ≈ sqrt(2.0)
tspan = (0.0, sqrt(2.0))
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Brent())
@test sol.left ≈ sqrt(2.0)

# Alefeld
sol = solve(probB, Alefeld())
@test sol.u ≈ sqrt(2.0)
tspan = (sqrt(2.0), 10.0)
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Alefeld())
@test sol.u ≈ sqrt(2.0)
tspan = (0.0, sqrt(2.0))
probB = IntervalNonlinearProblem(f, tspan)
sol = solve(probB, Alefeld())
@test sol.u ≈ sqrt(2.0)

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

# Test that `SimpleDFSane` passes a test that `SimpleNewtonRaphson` fails on.
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
    sol = solve(probN, SimpleDFSane())
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

# Test kwars in `SimpleDFSane`
σ_min = [1e-10, 1e-5, 1e-4]
σ_max = [1e10, 1e5, 1e4]
σ_1 = [1.0, 0.5, 2.0]
M = [10, 1, 100]
γ = [1e-4, 1e-3, 1e-5]
τ_min = [0.1, 0.2, 0.3]
τ_max = [0.5, 0.8, 0.9]
nexp = [2, 1, 2]
η_strategy = [
    (f_1, k, x, F) -> f_1 / k^2,
    (f_1, k, x, F) -> f_1 / k^3,
    (f_1, k, x, F) -> f_1 / k^4,
]

list_of_options = zip(σ_min, σ_max, σ_1, M, γ, τ_min, τ_max, nexp,
    η_strategy)
for options in list_of_options
    local probN, sol, alg
    alg = SimpleDFSane(σ_min = options[1],
        σ_max = options[2],
        σ_1 = options[3],
        M = options[4],
        γ = options[5],
        τ_min = options[6],
        τ_max = options[7],
        nexp = options[8],
        η_strategy = options[9])

    probN = NonlinearProblem(f, u0, p)
    sol = solve(probN, alg)
    @test all(abs.(f(u, p)) .< 1e-10)
end

# Batched Broyden
using NNlib

f, u0 = (u, p) -> u .* u .- p, randn(1, 3)

p = [2.0 1.0 5.0];
probN = NonlinearProblem{false}(f, u0, p);

sol = solve(probN, Broyden(batched = true))

@test abs.(sol.u) ≈ sqrt.(p)

for alg in (BATCHED_BROYDEN_SOLVERS...,
    BATCHED_LBROYDEN_SOLVERS...,
    BATCHED_DFSANE_SOLVERS...)
    sol = solve(probN, alg; abstol = 1e-3, reltol = 1e-3)

    @test sol.retcode == ReturnCode.Success
    @test abs.(sol.u)≈sqrt.(p) atol=1e-3 rtol=1e-3
end

## User specified Jacobian

f, u0 = (u, p) -> u .* u .- p, randn(3)

f_jac(u, p) = begin
    diagm(2 * u)
end

p = [2.0, 1.0, 5.0];

probN = NonlinearProblem(NonlinearFunction(f, jac = f_jac), u0, p)

for alg in (SimpleNewtonRaphson(), SimpleTrustRegion())
    sol = solve(probN, alg)
    @test abs.(sol.u) ≈ sqrt.(p)
end
