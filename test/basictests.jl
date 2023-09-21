using BenchmarkTools, LinearSolve, NonlinearSolve, StaticArrays, Random, LinearAlgebra,
    Test, ForwardDiff, Zygote, Enzyme, SparseDiffTools

_nameof(x) = applicable(nameof, x) ? nameof(x) : _nameof(typeof(x))

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)
quadratic_f2(u, p) = @. p[1] * u * u - p[2]

function newton_fails(u, p)
    return 0.010000000000000002 .+
           10.000000000000002 ./ (1 .+
            (0.21640425613334457 .+
             216.40425613334457 ./ (1 .+
              (0.21640425613334457 .+
               216.40425613334457 ./
               (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^ 2.0) .-
           0.0011552453009332421u .- p
end

# --- NewtonRaphson tests ---

@testset "NewtonRaphson" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; linesearch = LineSearch())
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, NewtonRaphson(; linesearch), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linsolve, precs,
        linesearch = LineSearch())
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, NewtonRaphson(; linsolve, precs, linesearch), abstol = 1e-9)
    end

    @testset "LineSearch: $(_nameof(lsmethod)) LineSearch AD: $(_nameof(ad))" for lsmethod in (Static(),
            StrongWolfe(), BackTracking(), HagerZhang(), MoreThuente()),
        ad in (AutoFiniteDiff(), AutoZygote())

        linesearch = LineSearch(; method = lsmethod, autodiff = ad)
        u0s = VERSION ≥ v"1.9" ? ([1.0, 1.0], @SVector[1.0, 1.0], 1.0) : ([1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0; linesearch)
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
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linsolve, precs = prec, linesearch)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                NewtonRaphson(; linsolve, precs = prec), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    if VERSION ≥ v"1.9"
        @testset "[OOP] [Immutable AD] p: $(p)" for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
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

    if VERSION ≥ v"1.9"
        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]
        @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
        @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
            p) ≈
              ForwardDiff.jacobian(t, p)
    end

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

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (false, true,
            AutoSparseForwardDiff(), AutoSparseFiniteDiff(), AutoZygote(),
            AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(; autodiff)).u .≈ sqrt(2.0))
    end
end

# --- TrustRegion tests ---

@testset "TrustRegion" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; radius_update_scheme, kwargs...)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, TrustRegion(; radius_update_scheme); abstol = 1e-9, kwargs...)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; radius_update_scheme, kwargs...)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, TrustRegion(; radius_update_scheme); abstol = 1e-9, kwargs...)
    end

    radius_update_schemes = [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
    u0s = VERSION ≥ v"1.9" ? ([1.0, 1.0], @SVector[1.0, 1.0], 1.0) : ([1.0, 1.0], 1.0)

    @testset "[OOP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme)" for u0 in u0s,
        radius_update_scheme in radius_update_schemes

        sol = benchmark_nlsolve_oop(quadratic_f, u0; radius_update_scheme)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme)" for u0 in ([
            1.0, 1.0],), radius_update_scheme in radius_update_schemes
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; radius_update_scheme)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    if VERSION ≥ v"1.9"
        @testset "[OOP] [Immutable AD] radius_update_scheme: $(radius_update_scheme) p: $(p)" for radius_update_scheme in radius_update_schemes,
            p in 1.0:0.1:100.0

            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p;
                    radius_update_scheme)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p; radius_update_scheme).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] [Scalar AD] radius_update_scheme: $(radius_update_scheme)  p: $(p)" for radius_update_scheme in radius_update_schemes,
        p in 1.0:0.1:100.0

        @test begin
            res = benchmark_nlsolve_oop(quadratic_f, oftype(p, 1.0), p;
                radius_update_scheme)
            res_true = sqrt(p)
            res.u ≈ res_true
        end
        @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, oftype(p, 1.0),
                p; radius_update_scheme).u, p) ≈ 1 / (2 * sqrt(p))
    end

    if VERSION ≥ v"1.9"
        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]
        @testset "[OOP] [Jacobian] radius_update_scheme: $(radius_update_scheme)" for radius_update_scheme in radius_update_schemes
            @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p; radius_update_scheme).u ≈
                  sqrt(p[2] / p[1])
            @test ForwardDiff.jacobian(p -> [
                    benchmark_nlsolve_oop(quadratic_f2, 0.5, p;
                        radius_update_scheme).u,
                ], p) ≈ ForwardDiff.jacobian(t, p)
        end
    end

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, TrustRegion(); maxiters = 100, abstol = 1e-10)
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

    @testset "ADType: $(autodiff) u0: $(_nameof(u0)) radius_update_scheme: $(radius_update_scheme)" for autodiff in (false,
            true, AutoSparseForwardDiff(), AutoSparseFiniteDiff(), AutoZygote(),
            AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0]),
        radius_update_scheme in radius_update_schemes

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(; autodiff)).u .≈ sqrt(2.0))
    end

    # Test that `TrustRegion` passes a test that `NewtonRaphson` fails on.
    @testset "Newton Raphson Fails: radius_update_scheme: $(radius_update_scheme)" for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sol = benchmark_nlsolve_oop(newton_fails, u0, p; radius_update_scheme)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
    end

    # Test kwargs in `TrustRegion`
    @testset "Keyword Arguments" begin
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
                initial_trust_radius = options[2], step_threshold = options[3],
                shrink_threshold = options[4], expand_threshold = options[5],
                shrink_factor = options[6], expand_factor = options[7],
                max_shrink_times = options[8])

            probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
            sol = solve(probN, alg, abstol = 1e-10)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end

    # Testing consistency of iip vs oop iterations
    @testset "OOP / IIP Consistency" begin
        maxiterations = [2, 3, 4, 5]
        u0 = [1.0, 1.0]
        @testset "radius_update_scheme: $(radius_update_scheme) maxiters: $(maxiters)" for radius_update_scheme in radius_update_schemes,
            maxiters in maxiterations

            sol_iip = benchmark_nlsolve_iip(quadratic_f!, u0; radius_update_scheme,
                maxiters)
            sol_oop = benchmark_nlsolve_oop(quadratic_f, u0; radius_update_scheme,
                maxiters)
            @test sol_iip.u ≈ sol_iip.u
        end
    end
end

# --- LevenbergMarquardt tests ---

@testset "LevenbergMarquardt" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, LevenbergMarquardt(), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, LevenbergMarquardt(), abstol = 1e-9)
    end

    u0s = VERSION ≥ v"1.9" ? ([1.0, 1.0], @SVector[1.0, 1.0], 1.0) : ([1.0, 1.0], 1.0)
    @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
        sol = benchmark_nlsolve_oop(quadratic_f, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), LevenbergMarquardt(),
            abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0), LevenbergMarquardt(),
            abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    if VERSION ≥ v"1.9"
        @testset "[OOP] [Immutable AD] p: $(p)" for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
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

    if VERSION ≥ v"1.9"
        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]
        @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
        @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
            p) ≈
              ForwardDiff.jacobian(t, p)
    end

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (false, true,
            AutoSparseForwardDiff(), AutoSparseFiniteDiff(), AutoZygote(),
            AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, LevenbergMarquardt(; autodiff)).u .≈ sqrt(2.0))
    end

    # Test that `LevenbergMarquardt` passes a test that `NewtonRaphson` fails on.
    @testset "Newton Raphson Fails" begin
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sol = benchmark_nlsolve_oop(newton_fails, u0, p)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
    end

    # Test kwargs in `LevenbergMarquardt`
    @testset "Keyword Arguments" begin
        damping_initial = [0.5, 2.0, 5.0]
        damping_increase_factor = [1.5, 3.0, 10.0]
        damping_decrease_factor = Float64[2, 5, 10]
        finite_diff_step_geodesic = [0.02, 0.2, 0.3]
        α_geodesic = [0.6, 0.8, 0.9]
        b_uphill = Float64[0, 1, 2]
        min_damping_D = [1e-12, 1e-9, 1e-4]

        list_of_options = zip(damping_initial, damping_increase_factor,
            damping_decrease_factor, finite_diff_step_geodesic, α_geodesic, b_uphill,
            min_damping_D)
        for options in list_of_options
            local probN, sol, alg
            alg = LevenbergMarquardt(damping_initial = options[1],
                damping_increase_factor = options[2],
                damping_decrease_factor = options[3],
                finite_diff_step_geodesic = options[4], α_geodesic = options[5],
                b_uphill = options[6], min_damping_D = options[7])

            probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
            sol = solve(probN, alg, abstol = 1e-10)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end
end
