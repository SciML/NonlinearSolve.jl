using BenchmarkTools, LinearSolve, NonlinearSolve, StaticArrays, Random, LinearAlgebra,
    Test, ForwardDiff, Zygote, Enzyme, SparseDiffTools

_nameof(x) = applicable(nameof, x) ? nameof(x) : _nameof(typeof(x))

# --- NewtonRaphson tests ---

@testset "NewtonRaphson" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0)
        prob = NonlinearProblem{false}(f, u0, p)
        cache = init(prob, NewtonRaphson(), abstol = 1e-9)
        return solve!(cache)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linsolve, precs)
        prob = NonlinearProblem{true}(f, u0, p)
        cache = init(prob, NewtonRaphson(; linsolve, precs), abstol = 1e-9)
        return solve!(cache)
    end

    quadratic_f(u, p) = u .* u .- p
    quadratic_f!(du, u, p) = (du .= u .* u .- p)

    @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
        sol = benchmark_nlsolve_oop(quadratic_f, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), NewtonRaphson(),
            abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    precs = [NonlinearSolve.DEFAULT_PRECS, :Random]

    @testset "[IIP] u0: $(typeof(u0)) precs: $(_nameof(prec)) linsolve: $(_nameof(linsolve))" for u0 in ([
            1.0, 1.0],), prec in precs, linsolve in (nothing, KrylovJL_GMRES())
        if prec === :Random
            prec = (args...) -> (Diagonal(randn!(similar(u0))), nothing)
        end
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; linsolve, precs = prec)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            NewtonRaphson(; linsolve, precs = prec), abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    # Immutable
    @testset "[OOP] [Immutable AD] p: $(p)" for p in 1.0:0.1:100.0
        @test begin
            res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
            res_true = sqrt(p)
            all(res.u .≈ res_true)
        end
        @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
            @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
    end

    @testset "[OOP] [Scalar AD] p: $(p)" for p in 1.0:0.1:100.0
        @test begin
            res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
            res_true = sqrt(p)
            res.u ≈ res_true
        end
        @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u, p) ≈
              1 / (2 * sqrt(p))
    end

    quadratic_f2(u, p) = @. p[1] * u * u - p[2]
    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u], p) ≈
          ForwardDiff.jacobian(t, p)

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, NewtonRaphson(); maxiters = 100, abstol = 1e-10)
        sols = zeros(length(p_range))
        for (i, p) in enumerate(p_range)
            reinit!(cache, iip ? [cache.u[1]] : cache.u; p = p)
            sol = solve!(cache)
            sols[i] = iip ? sol.u[1] : sol.u
        end
        return sols
    end
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, Val(false)) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, Val(true)) ≈ sqrt.(p)

    probN = NonlinearProblem(quadratic_f, @SVector[1.0, 1.0], 2.0)
    @testset "ADType: $(autodiff) u0: $(u0)" for autodiff in (false, true,
        AutoSparseForwardDiff(), AutoSparseFiniteDiff(), AutoZygote(), AutoSparseZygote(),
        AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(; autodiff)).u .≈ sqrt(2.0))
    end
end

# --- TrustRegion tests ---
@testset "TrustRegion" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; radius_update_scheme)
        prob = NonlinearProblem{false}(f, u0, p)
        cache = init(prob, TrustRegion(; radius_update_scheme), abstol = 1e-9)
        return solve!(cache)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; radius_update_scheme)
        prob = NonlinearProblem{true}(f, u0, p)
        cache = init(prob, TrustRegion(; radius_update_scheme), abstol = 1e-9)
        return solve!(cache)
    end

    quadratic_f(u, p) = u .* u .- p
    quadratic_f!(du, u, p) = (du .= u .* u .- p)

    radius_update_schemes = [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]

    @testset "[OOP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme)" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0), radius_update_scheme in radius_update_schemes
        sol = benchmark_nlsolve_oop(quadratic_f, u0; radius_update_scheme)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme)" for u0 in ([1.0, 1.0],), radius_update_scheme in radius_update_schemes
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; radius_update_scheme)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end
end


# # Immutable
# f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, TrustRegion(), abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei),
#         abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan),
#         abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
#         abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
#         abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # Scalar
# f, u0 = (u, p) -> u * u - p, 1.0

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, TrustRegion(), abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei),
#         abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan),
#         abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
#         abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
#         abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# f = (u, p) -> p[1] * u * u - p[2]
# t = (p) -> [sqrt(p[2] / p[1])]
# p = [0.9, 50.0]
# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, TrustRegion())
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei))
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan))
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan))
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin))
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# # Iterator interface
# f = (u, p) -> u * u - p
# g = function (p_range)
#     probN = NonlinearProblem{false}(f, 0.5, p_range[begin])
#     cache = init(probN, TrustRegion(); maxiters = 100, abstol = 1e-10)
#     sols = zeros(length(p_range))
#     for (i, p) in enumerate(p_range)
#         reinit!(cache, cache.u; p = p)
#         sol = solve!(cache)
#         sols[i] = sol.u
#     end
#     return sols
# end
# p = range(0.01, 2, length = 200)
# @test g(p) ≈ sqrt.(p)

# f = (res, u, p) -> (res[begin] = u[1] * u[1] - p)
# g = function (p_range)
#     probN = NonlinearProblem{true}(f, [0.5], p_range[begin])
#     cache = init(probN, TrustRegion(); maxiters = 100, abstol = 1e-10)
#     sols = zeros(length(p_range))
#     for (i, p) in enumerate(p_range)
#         reinit!(cache, [cache.u[1]]; p = p)
#         sol = solve!(cache)
#         sols[i] = sol.u[1]
#     end
#     return sols
# end
# p = range(0.01, 2, length = 200)
# @test g(p) ≈ sqrt.(p)

# # Error Checks
# f, u0 = (u, p) -> u .* u .- 2, @SVector[1.0, 1.0]
# probN = NonlinearProblem(f, u0)

# @test solve(probN, TrustRegion()).u[end] ≈ sqrt(2.0)
# @test solve(probN, TrustRegion(; autodiff = false)).u[end] ≈ sqrt(2.0)

# @test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei)).u[end] ≈
#       sqrt(2.0)
# @test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei, autodiff = false)).u[end] ≈
#       sqrt(2.0)

# @test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan)).u[end] ≈
#       sqrt(2.0)
# @test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan, autodiff = false)).u[end] ≈
#       sqrt(2.0)

# @test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan)).u[end] ≈
#       sqrt(2.0)
# @test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan, autodiff = false)).u[end] ≈
#       sqrt(2.0)

# @test solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin)).u[end] ≈
#       sqrt(2.0)
# @test solve(probN, TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff = false)).u[end] ≈
#       sqrt(2.0)

# for u0 in [1.0, [1, 1.0]]
#     local f, probN, sol
#     f = (u, p) -> u .* u .- 2.0
#     probN = NonlinearProblem(f, u0)
#     sol = sqrt(2) * u0

#     @test solve(probN, TrustRegion()).u ≈ sol
#     @test solve(probN, TrustRegion()).u ≈ sol
#     @test solve(probN, TrustRegion(; autodiff = false)).u ≈ sol
# end

# # Test that `TrustRegion` passes a test that `NewtonRaphson` fails on.
# u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
# global g, f
# f = (u, p) -> 0.010000000000000002 .+
#               10.000000000000002 ./ (1 .+
#                (0.21640425613334457 .+
#                 216.40425613334457 ./ (1 .+
#                  (0.21640425613334457 .+
#                   216.40425613334457 ./
#                   (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^ 2.0) .-
#               0.0011552453009332421u .- p
# g = function (p)
#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, TrustRegion(), abstol = 1e-10)
#     return sol.u
# end
# p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# u = g(p)
# f(u, p)
# @test all(abs.(f(u, p)) .< 1e-10)

# g = function (p)
#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Fan),
#         abstol = 1e-10)
#     return sol.u
# end
# p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# u = g(p)
# f(u, p)
# @test all(abs.(f(u, p)) .< 1e-10)

# g = function (p)
#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin),
#         abstol = 1e-10)
#     return sol.u
# end
# p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# u = g(p)
# f(u, p)
# @test all(abs.(f(u, p)) .< 1e-10)

# # Test kwars in `TrustRegion`
# max_trust_radius = [10.0, 100.0, 1000.0]
# initial_trust_radius = [10.0, 1.0, 0.1]
# step_threshold = [0.0, 0.01, 0.25]
# shrink_threshold = [0.25, 0.3, 0.5]
# expand_threshold = [0.5, 0.8, 0.9]
# shrink_factor = [0.1, 0.3, 0.5]
# expand_factor = [1.5, 2.0, 3.0]
# max_shrink_times = [10, 20, 30]

# list_of_options = zip(max_trust_radius, initial_trust_radius, step_threshold,
#     shrink_threshold, expand_threshold, shrink_factor,
#     expand_factor, max_shrink_times)
# for options in list_of_options
#     local probN, sol, alg
#     alg = TrustRegion(max_trust_radius = options[1],
#         initial_trust_radius = options[2],
#         step_threshold = options[3],
#         shrink_threshold = options[4],
#         expand_threshold = options[5],
#         shrink_factor = options[6],
#         expand_factor = options[7],
#         max_shrink_times = options[8])

#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, alg, abstol = 1e-10)
#     @test all(abs.(f(u, p)) .< 1e-10)
# end

# # Testing consistency of iip vs oop iterations

# maxiterations = [2, 3, 4, 5]
# u0 = [1.0, 1.0]
# function iip_oop(f, fip, u0, radius_update_scheme, maxiters)
#     prob_iip = NonlinearProblem{true}(fip, u0)
#     solver = init(prob_iip, TrustRegion(radius_update_scheme = radius_update_scheme),
#         abstol = 1e-9, maxiters = maxiters)
#     sol_iip = solve!(solver)

#     prob_oop = NonlinearProblem{false}(f, u0)
#     solver = init(prob_oop, TrustRegion(radius_update_scheme = radius_update_scheme),
#         abstol = 1e-9, maxiters = maxiters)
#     sol_oop = solve!(solver)

#     return sol_iip.u[end], sol_oop.u[end]
# end

# for maxiters in maxiterations
#     iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Simple, maxiters)
#     @test iip == oop
# end

# for maxiters in maxiterations
#     iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Hei, maxiters)
#     @test iip == oop
# end

# for maxiters in maxiterations
#     iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Yuan, maxiters)
#     @test iip == oop
# end

# for maxiters in maxiterations
#     iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Fan, maxiters)
#     @test iip == oop
# end

# for maxiters in maxiterations
#     iip, oop = iip_oop(ff, ffiip, u0, RadiusUpdateSchemes.Bastin, maxiters)
#     @test iip == oop
# end

# # --- LevenbergMarquardt tests ---

# function benchmark_immutable(f, u0)
#     probN = NonlinearProblem{false}(f, u0)
#     solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
#     sol = solve!(solver)
# end

# function benchmark_mutable(f, u0)
#     probN = NonlinearProblem{false}(f, u0)
#     solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
#     sol = solve!(solver)
# end

# function benchmark_scalar(f, u0)
#     probN = NonlinearProblem{false}(f, u0)
#     sol = (solve(probN, LevenbergMarquardt(), abstol = 1e-9))
# end

# function ff(u, p)
#     u .* u .- 2
# end

# function sf(u, p)
#     u * u - 2
# end
# u0 = [1.0, 1.0]

# sol = benchmark_immutable(ff, cu0)
# @test SciMLBase.successful_retcode(sol)
# @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
# sol = benchmark_mutable(ff, u0)
# @test SciMLBase.successful_retcode(sol)
# @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
# sol = benchmark_scalar(sf, csu0)
# @test SciMLBase.successful_retcode(sol)
# @test abs(sol.u * sol.u - 2) < 1e-9

# function benchmark_inplace(f, u0)
#     probN = NonlinearProblem{true}(f, u0)
#     solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
#     sol = solve!(solver)
# end

# function ffiip(du, u, p)
#     du .= u .* u .- 2
# end
# u0 = [1.0, 1.0]

# sol = benchmark_inplace(ffiip, u0)
# @test SciMLBase.successful_retcode(sol)
# @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

# u0 = [1.0, 1.0]
# probN = NonlinearProblem{true}(ffiip, u0)
# solver = init(probN, LevenbergMarquardt(), abstol = 1e-9)
# @test (@ballocated solve!(solver)) < 120

# # AD Tests
# using ForwardDiff

# # Immutable
# f, u0 = (u, p) -> u .* u .- p, @SVector[1.0, 1.0]

# g = function (p)
#     probN = NonlinearProblem{false}(f, csu0, p)
#     sol = solve(probN, LevenbergMarquardt(), abstol = 1e-9)
#     return sol.u[end]
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # Scalar
# f, u0 = (u, p) -> u * u - p, 1.0

# g = function (p)
#     probN = NonlinearProblem{false}(f, oftype(p, u0), p)
#     sol = solve(probN, LevenbergMarquardt(), abstol = 1e-10)
#     return sol.u
# end

# @test ForwardDiff.derivative(g, 3.0) ≈ 1 / (2 * sqrt(3.0))

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# f = (u, p) -> p[1] * u * u - p[2]
# t = (p) -> [sqrt(p[2] / p[1])]
# p = [0.9, 50.0]
# gnewton = function (p)
#     probN = NonlinearProblem{false}(f, 0.5, p)
#     sol = solve(probN, LevenbergMarquardt())
#     return [sol.u]
# end
# @test gnewton(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(gnewton, p) ≈ ForwardDiff.jacobian(t, p)

# # Error Checks
# f, u0 = (u, p) -> u .* u .- 2.0, @SVector[1.0, 1.0]
# probN = NonlinearProblem(f, u0)

# @test solve(probN, LevenbergMarquardt()).u[end] ≈ sqrt(2.0)
# @test solve(probN, LevenbergMarquardt(; autodiff = false)).u[end] ≈ sqrt(2.0)

# for u0 in [1.0, [1, 1.0]]
#     local f, probN, sol
#     f = (u, p) -> u .* u .- 2.0
#     probN = NonlinearProblem(f, u0)
#     sol = sqrt(2) * u0

#     @test solve(probN, LevenbergMarquardt()).u ≈ sol
#     @test solve(probN, LevenbergMarquardt()).u ≈ sol
#     @test solve(probN, LevenbergMarquardt(; autodiff = false)).u ≈ sol
# end

# # Test that `LevenbergMarquardt` passes a test that `NewtonRaphson` fails on.
# u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
# global g, f
# f = (u, p) -> 0.010000000000000002 .+
#               10.000000000000002 ./ (1 .+
#                (0.21640425613334457 .+
#                 216.40425613334457 ./ (1 .+
#                  (0.21640425613334457 .+
#                   216.40425613334457 ./
#                   (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^ 2.0) .-
#               0.0011552453009332421u .- p
# g = function (p)
#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, LevenbergMarquardt(), abstol = 1e-10)
#     return sol.u
# end
# p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# u = g(p)
# f(u, p)
# @test all(abs.(f(u, p)) .< 1e-10)

# # # Test kwars in `LevenbergMarquardt`
# damping_initial = [0.5, 2.0, 5.0]
# damping_increase_factor = [1.5, 3.0, 10.0]
# damping_decrease_factor = [2, 5, 10]
# finite_diff_step_geodesic = [0.02, 0.2, 0.3]
# α_geodesic = [0.6, 0.8, 0.9]
# b_uphill = [0, 1, 2]
# min_damping_D = [1e-12, 1e-9, 1e-4]

# list_of_options = zip(damping_initial, damping_increase_factor, damping_decrease_factor,
#     finite_diff_step_geodesic, α_geodesic, b_uphill,
#     min_damping_D)
# for options in list_of_options
#     local probN, sol, alg
#     alg = LevenbergMarquardt(damping_initial = options[1],
#         damping_increase_factor = options[2],
#         damping_decrease_factor = options[3],
#         finite_diff_step_geodesic = options[4],
#         α_geodesic = options[5],
#         b_uphill = options[6],
#         min_damping_D = options[7])

#     probN = NonlinearProblem{false}(f, u0, p)
#     sol = solve(probN, alg, abstol = 1e-10)
#     @test all(abs.(f(u, p)) .< 1e-10)
# end
