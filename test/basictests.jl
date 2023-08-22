using NonlinearSolve
using StaticArrays
using BenchmarkTools
using LinearSolve
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
    sol = (solve(probN, NewtonRaphson(), abstol = 1e-9))
end

function ff(u, p)
    u .* u .- 2
end
const cu0 = @SVector[1.0, 1.0]
function sf(u, p)
    u * u - 2
end
const csu0 = 1.0
u0 = [1.0, 1.0]

sol = benchmark_immutable(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
sol = benchmark_mutable(ff, u0)
@test sol.retcode === ReturnCode.Success
@test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test abs(sol.u * sol.u - 2) < 1e-9

# @test (@ballocated benchmark_immutable(ff, cu0)) < 200
# @test (@ballocated benchmark_mutable(ff, cu0)) < 200
# @test (@ballocated benchmark_scalar(sf, csu0)) < 400

function benchmark_inplace(f, u0, linsolve, precs)
    probN = NonlinearProblem{true}(f, u0)
    solver = init(probN, NewtonRaphson(; linsolve, precs), abstol = 1e-9)
    sol = solve!(solver)
end

function ffiip(du, u, p)
    du .= u .* u .- 2
end
u0 = [1.0, 1.0]

precs = [
    NonlinearSolve.DEFAULT_PRECS,
    (args...) -> (Diagonal(rand!(similar(u0))), nothing)
]

for prec in precs, linsolve in (nothing, KrylovJL_GMRES())
    sol = benchmark_inplace(ffiip, u0, linsolve, prec)
    @test sol.retcode === ReturnCode.Success
    @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
end

u0 = [1.0, 1.0]
probN = NonlinearProblem{true}(ffiip, u0)
solver = init(probN, NewtonRaphson(), abstol = 1e-9)
@test (@ballocated solve!(solver)) <= 64

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

# Iterator interface
f = (u, p) -> u * u - p
g = function (p_range)
    probN = NonlinearProblem{false}(f, 0.5, p_range[begin])
    cache = init(probN, NewtonRaphson(); maxiters = 100, abstol = 1e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, cache.u; p = p)
        sol = solve!(cache)
        sols[i] = sol.u
    end
    return sols
end
p = range(0.01, 2, length = 200)
@test g(p) ≈ sqrt.(p)

f = (res, u, p) -> (res[begin] = u[1] * u[1] - p)
g = function (p_range)
    probN = NonlinearProblem{true}(f, [0.5], p_range[begin])
    cache = init(probN, NewtonRaphson(); maxiters = 100, abstol = 1e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, [cache.u[1]]; p = p)
        sol = solve!(cache)
        sols[i] = sol.u[1]
    end
    return sols
end
p = range(0.01, 2, length = 200)
@test g(p) ≈ sqrt.(p)

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

function benchmark_immutable(f, u0, radius_update_scheme)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_mutable(f, u0, radius_update_scheme)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_scalar(f, u0, radius_update_scheme)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9))
end

function ff(u, p = nothing)
    u .* u .- 2
end

function sf(u, p = nothing)
    u * u - 2
end

u0 = [1.0, 1.0]
radius_update_schemes = [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.Hei,
    RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]

for radius_update_scheme in radius_update_schemes
    sol = benchmark_immutable(ff, cu0, radius_update_scheme)
    @test sol.retcode === ReturnCode.Success
    @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
    sol = benchmark_mutable(ff, u0, radius_update_scheme)
    @test sol.retcode === ReturnCode.Success
    @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
    sol = benchmark_scalar(sf, csu0, radius_update_scheme)
    @test sol.retcode === ReturnCode.Success
    @test abs(sol.u * sol.u - 2) < 1e-9
end

function benchmark_inplace(f, u0, radius_update_scheme)
    probN = NonlinearProblem{true}(f, u0)
    solver = init(probN, TrustRegion(; radius_update_scheme), abstol = 1e-9)
    sol = solve!(solver)
end

function ffiip(du, u, p = nothing)
    du .= u .* u .- 2
end
u0 = [1.0, 1.0]

for radius_update_scheme in radius_update_schemes
    sol = benchmark_inplace(ffiip, u0, radius_update_scheme)
    @test sol.retcode === ReturnCode.Success
    @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
end

for radius_update_scheme in radius_update_schemes
    probN = NonlinearProblem{true}(ffiip, u0)
    solver = init(probN, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9)
    @test (@ballocated solve!(solver)) < 200
end

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, TrustRegion(), abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei),
        abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan),
        abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
        abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
        abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(), abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei),
        abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan),
        abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
        abstol = 1e-10)
    return sol.u
end

@test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
        abstol = 1e-10)
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
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, TrustRegion())
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

gnewton = function (p)
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei))
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

gnewton = function (p)
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan))
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

gnewton = function (p)
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan))
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

gnewton = function (p)
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin))
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# Iterator interface
f = (u, p) -> u * u - p
g = function (p_range)
    probN = NonlinearProblem{false}(f, 0.5, p_range[begin])
    cache = init(probN, TrustRegion(); maxiters = 100, abstol = 1e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, cache.u; p = p)
        sol = solve!(cache)
        sols[i] = sol.u
    end
    return sols
end
p = range(0.01, 2, length = 200)
@test g(p) ≈ sqrt.(p)

f = (res, u, p) -> (res[begin] = u[1] * u[1] - p)
g = function (p_range)
    probN = NonlinearProblem{true}(f, [0.5], p_range[begin])
    cache = init(probN, TrustRegion(); maxiters = 100, abstol = 1e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, [cache.u[1]]; p = p)
        sol = solve!(cache)
        sols[i] = sol.u[1]
    end
    return sols
end
p = range(0.01, 2, length = 200)
@test g(p) ≈ sqrt.(p)

# Error Checks
f, u0 = (u, p) -> u .* u .- 2, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, TrustRegion()).u[end] ≈ sqrt(2.0)
@test solve(probN, TrustRegion(; autodiff = false)).u[end] ≈ sqrt(2.0)

@test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei)).u[end] ≈
      sqrt(2.0)
@test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei, autodiff = false)).u[end] ≈
      sqrt(2.0)

@test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan)).u[end] ≈
      sqrt(2.0)
@test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan, autodiff = false)).u[end] ≈
      sqrt(2.0)

@test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan)).u[end] ≈
      sqrt(2.0)
@test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan, autodiff = false)).u[end] ≈
      sqrt(2.0)

@test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin)).u[end] ≈
      sqrt(2.0)
@test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff = false)).u[end] ≈
      sqrt(2.0)

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
    sol = solve(probN, TrustRegion(), abstol = 1e-10)
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

g = function (p)
    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
        abstol = 1e-10)
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

g = function (p)
    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
        abstol = 1e-10)
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

    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, alg, abstol = 1e-10)
    @test all(abs.(f(u, p)) .< 1e-10)
end

# Testing consistency of iip vs oop iterations

maxiterations = [2, 3, 4, 5]
u0 = [1.0, 1.0]
function iip_oop(f, fip, u0, radius_update_scheme, maxiters)
    prob_iip = NonlinearProblem{true}(fip, u0)
    solver = init(prob_iip, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9, maxiters = maxiters)
    sol_iip = solve!(solver)

    prob_oop = NonlinearProblem{false}(f, u0)
    solver = init(prob_oop, TrustRegion(radius_update_scheme = radius_update_scheme),
        abstol = 1e-9, maxiters = maxiters)
    sol_oop = solve!(solver)

    return sol_iip.u[end], sol_oop.u[end]
end

for maxiters in maxiterations
    iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Simple, maxiters)
    @test iip == oop
end

for maxiters in maxiterations
    iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Hei, maxiters)
    @test iip == oop
end

for maxiters in maxiterations
    iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Yuan, maxiters)
    @test iip == oop
end

for maxiters in maxiterations
    iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Fan, maxiters)
    @test iip == oop
end

for maxiters in maxiterations
    iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Bastin, maxiters)
    @test iip == oop
end

# --- LevenbergMarquardt tests ---

function benchmark_immutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_mutable(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
    sol = solve!(solver)
end

function benchmark_scalar(f, u0)
    probN = NonlinearProblem{false}(f, u0)
    sol = (solve(probN, LevenbergMarquardt(), abstol = 1e-9))
end

function ff(u, p)
    u .* u .- 2
end

function sf(u, p)
    u * u - 2
end
u0 = [1.0, 1.0]

sol = benchmark_immutable(ff, cu0)
@test sol.retcode === ReturnCode.Success
@test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
sol = benchmark_mutable(ff, u0)
@test sol.retcode === ReturnCode.Success
@test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
sol = benchmark_scalar(sf, csu0)
@test sol.retcode === ReturnCode.Success
@test abs(sol.u * sol.u - 2) < 1e-9

function benchmark_inplace(f, u0)
    probN = NonlinearProblem{true}(f, u0)
    solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
    sol = solve!(solver)
end

function ffiip(du, u, p)
    du .= u .* u .- 2
end
u0 = [1.0, 1.0]

sol = benchmark_inplace(ffiip, u0)
@test sol.retcode === ReturnCode.Success
@test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

u0 = [1.0, 1.0]
probN = NonlinearProblem{true}(ffiip, u0)
solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
@test (@ballocated solve!(solver)) < 120

# AD Tests
using ForwardDiff

# Immutable
f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

g = function (p)
    probN = NonlinearProblem{false}(f, csu0, p)
    sol = solve(probN, LevenbergMarquardt(), abstol = 1e-9)
    return sol.u[end]
end

for p in 1.1:0.1:100.0
    @test g(p) ≈ sqrt(p)
    @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
end

# Scalar
f, u0 = (u, p) -> u * u - p, 1.0

g = function (p)
    probN = NonlinearProblem{false}(f, oftype(p, u0), p)
    sol = solve(probN, LevenbergMarquardt(), abstol = 1e-10)
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
    probN = NonlinearProblem{false}(f, 0.5, p)
    sol = solve(probN, LevenbergMarquardt())
    return [sol.u]
end
@test gnewton(p) ≈ [sqrt(p[2] / p[1])]
@test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# Error Checks
f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)

@test solve(probN, LevenbergMarquardt()).u[end] ≈ sqrt(2.0)
@test solve(probN, LevenbergMarquardt(; autodiff = false)).u[end] ≈ sqrt(2.0)

for u0 in [1.0, [1, 1.0]]
    local f, probN, sol
    f = (u, p) -> u .* u .- 2.0
    probN = NonlinearProblem(f, u0)
    sol = sqrt(2) * u0

    @test solve(probN, LevenbergMarquardt()).u ≈ sol
    @test solve(probN, LevenbergMarquardt()).u ≈ sol
    @test solve(probN, LevenbergMarquardt(; autodiff = false)).u ≈ sol
end

# Test that `LevenbergMarquardt` passes a test that `NewtonRaphson` fails on.
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
    sol = solve(probN, LevenbergMarquardt(), abstol = 1e-10)
    return sol.u
end
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u = g(p)
f(u, p)
@test all(abs.(f(u, p)) .< 1e-10)

# # Test kwars in `LevenbergMarquardt`
damping_initial = [0.5, 2.0, 5.0]
damping_increase_factor = [1.5, 3.0, 10.0]
damping_decrease_factor = [2, 5, 10]
finite_diff_step_geodesic = [0.02, 0.2, 0.3]
α_geodesic = [0.6, 0.8, 0.9]
b_uphill = [0, 1, 2]
min_damping_D = [1e-12, 1e-9, 1e-4]

list_of_options = zip(damping_initial, damping_increase_factor, damping_decrease_factor,
    finite_diff_step_geodesic, α_geodesic, b_uphill,
    min_damping_D)
for options in list_of_options
    local probN, sol, alg
    alg = LevenbergMarquardt(damping_initial = options[1],
        damping_increase_factor = options[2],
        damping_decrease_factor = options[3],
        finite_diff_step_geodesic = options[4],
        α_geodesic = options[5],
        b_uphill = options[6],
        min_damping_D = options[7])

    probN = NonlinearProblem{false}(f, u0, p)
    sol = solve(probN, alg, abstol = 1e-10)
    @test all(abs.(f(u, p)) .< 1e-10)
end
