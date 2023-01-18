using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

# --- NewtonRaphson tests ---

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
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9

# @test (@ballocated benchmark_immutable(ff, cu0)) < 200
# @test (@ballocated benchmark_mutable(ff, cu0)) < 200
# @test (@ballocated benchmark_scalar(sf, csu0)) < 400

function benchmark_inplace(f, u0)
    probN = NonlinearProblem{true}(f, u0)
    solver = init(probN, NewtonRaphson(), abstol = 1e-9)
    sol = solve!(solver)
end

function ffiip(du, u, p)
    du .= u .* u .- 2
end
u0 = [1.0, 1.0]

sol = benchmark_inplace(ffiip, u0)
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)

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

# --- TrustRegion tests ---

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, TrustRegion(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_mutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, TrustRegion(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, TrustRegion()))
end

function ff(u, p)
    u .* u .- 2
end

function sf(u, p)
    u * u - 2
end

sol = benchmark_immutable(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_mutable(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test sol.u * sol.u - 2 < 1e-9

@test (@ballocated benchmark_immutable(ff, cu0)) < 400
@test (@ballocated benchmark_mutable(ff, cu0)) < 400
@test (@ballocated benchmark_scalar(sf, csu0)) < 400

function benchmark_inplace(f, u0)
    probN = NonlinearProblem(f, u0)
    solver = init(probN, TrustRegion(), abstol = 1e-9)
    sol = solve!(solver)
end

function ffiip(du, u, p)
    du .= u .* u .- 2
end
u0 = [1.0, 1.0]

sol = benchmark_inplace(ffiip, u0)
@test sol.retcode === ReturnCode.Success
@test all(sol.u .* sol.u .- 2 .< 1e-9)

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem(f, csu0, p)
    sol = solve(probN, TrustRegion(), abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0

g = function (p)
    probN = NonlinearProblem(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(), abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

f = (u, p) -> p[1] * u * u - p[2]
t = (p) -> [sqrt(p[2] / p[1])]
p = [0.9, 50.0]
gnewton = function (p)
    probN = NonlinearProblem(f, 0.5, p)
    sol = solve(probN, TrustRegion())
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# Error Checks
f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, TrustRegion()).u[end] ≈ sqrt(2.0)
@test solve(probN, TrustRegion(; autodiff = false)).u[end] ≈ sqrt(2.0)

for u0 in [1.0, [1, 1.0]]
    local f, probN, sol
    f = (u, p) -> u .* u .- 2.0
    probN = NonlinearProblem(f, u0)
    sol = sqrt(2) * u0

    @test solve(probN, TrustRegion()).u ≈ sol
    @test solve(probN, TrustRegion()).u ≈ sol
    @test solve(probN, TrustRegion(; autodiff = false)).u ≈ sol
end

# Test that `TrustRegion` passes a test that `NewtonRaphson` fails on.
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
    sol = solve(probN, TrustRegion())
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

# Test kwars in `TrustRegion`
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
    alg = TrustRegion(max_trust_radius = options[1],
                      initial_trust_radius = options[2],
                      step_threshold = options[3],
                      shrink_threshold = options[4],
                      expand_threshold = options[5],
                      shrink_factor = options[6],
                      expand_factor = options[7],
                      max_shrink_times = options[8])

    probN = NonlinearProblem(f, u0, p)
    sol = solve(probN, alg)
    @test all(f(u, p) .< 1e-10)
end
