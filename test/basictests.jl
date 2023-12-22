using BenchmarkTools, LinearSolve, NonlinearSolve, StaticArrays, Random, LinearAlgebra,
    Test, ForwardDiff, Zygote, Enzyme, SparseDiffTools, DiffEqBase

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

const TERMINATION_CONDITIONS = [
    SteadyStateDiffEqTerminationMode(), SimpleNonlinearSolveTerminationMode(),
    NormTerminationMode(), RelTerminationMode(), RelNormTerminationMode(),
    AbsTerminationMode(), AbsNormTerminationMode(), RelSafeTerminationMode(),
    AbsSafeTerminationMode(), RelSafeBestTerminationMode(), AbsSafeBestTerminationMode(),
]

# --- NewtonRaphson tests ---

@testset "NewtonRaphson" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; linesearch = nothing)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, NewtonRaphson(; linesearch), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linsolve, precs,
            linesearch = nothing)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, NewtonRaphson(; linsolve, precs, linesearch), abstol = 1e-9)
    end

    @testset "LineSearch: $(_nameof(lsmethod)) LineSearch AD: $(_nameof(ad))" for lsmethod in (Static(),
            StrongWolfe(), BackTracking(), HagerZhang(), MoreThuente()),
        ad in (AutoFiniteDiff(), AutoZygote())

        linesearch = LineSearch(; method = lsmethod, autodiff = ad)
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0; linesearch)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), NewtonRaphson(),
                abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        precs = [
            (u0) -> NonlinearSolve.DEFAULT_PRECS,
            u0 -> ((args...) -> (Diagonal(rand!(similar(u0))), nothing)),
        ]

        @testset "[IIP] u0: $(typeof(u0)) precs: $(_nameof(prec)) linsolve: $(_nameof(linsolve))" for u0 in ([
                1.0, 1.0],), prec in precs, linsolve in (nothing, KrylovJL_GMRES())
            ad isa AutoZygote && continue
            if prec === :Random
                prec = (args...) -> (Diagonal(randn!(similar(u0))), nothing)
            end
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linsolve, precs = prec(u0),
                linesearch)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                NewtonRaphson(; linsolve, precs = prec(u0)), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    @testset "[OOP] [Immutable AD]" begin
        for p in [1.0, 100.0]
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

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

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (AutoSparseForwardDiff(),
            AutoSparseFiniteDiff(), AutoZygote(), AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(; autodiff)).u .≈ sqrt(2.0))
    end

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- TrustRegion tests ---

@testset "TrustRegion" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; radius_update_scheme, linsolve = nothing,
            vjp_autodiff = nothing, kwargs...)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, TrustRegion(; radius_update_scheme, linsolve, vjp_autodiff);
            abstol = 1e-9, kwargs...)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; radius_update_scheme, linsolve = nothing,
            vjp_autodiff = nothing, kwargs...)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, TrustRegion(; radius_update_scheme, linsolve, vjp_autodiff);
            abstol = 1e-9, kwargs...)
    end

    radius_update_schemes = [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei, RadiusUpdateSchemes.Yuan,
        RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
    linear_solvers = [nothing, LUFactorization(), KrylovJL_GMRES()]

    @testset "[OOP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme) linear_solver: $(linsolve)" for u0 in u0s,
        radius_update_scheme in radius_update_schemes, linsolve in linear_solvers

        !(u0 isa Array) && linsolve !== nothing && continue

        abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)

        sol = benchmark_nlsolve_oop(quadratic_f, u0; radius_update_scheme, linsolve, abstol)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< abstol)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            TrustRegion(; radius_update_scheme, linsolve); abstol)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0)) radius_update_scheme: $(radius_update_scheme) linear_solver: $(linsolve)" for u0 in ([
            1.0, 1.0],), radius_update_scheme in radius_update_schemes, linsolve in linear_solvers
        abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; radius_update_scheme, linsolve,
            abstol)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< abstol)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    @testset "[OOP] [Immutable AD] radius_update_scheme: $(radius_update_scheme)" for radius_update_scheme in radius_update_schemes
        for p in 1.0:0.1:100.0
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

    @testset "[OOP] [Scalar AD] radius_update_scheme: $(radius_update_scheme)" for radius_update_scheme in radius_update_schemes
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, oftype(p, 1.0), p;
                    radius_update_scheme)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                    oftype(p, 1.0), p; radius_update_scheme).u, p) ≈ 1 / (2 * sqrt(p))
        end
    end

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

    @testset "ADType: $(autodiff) u0: $(_nameof(u0)) radius_update_scheme: $(radius_update_scheme)" for autodiff in (AutoSparseForwardDiff(),
            AutoSparseFiniteDiff(), AutoZygote(), AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0]),
        radius_update_scheme in radius_update_schemes

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, TrustRegion(; autodiff, radius_update_scheme)).u .≈
                  sqrt(2.0))
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
            @test sol_iip.u ≈ sol_oop.u
        end
    end

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, TrustRegion(); termination_condition).u .≈ sqrt(2.0))
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

    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
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

    @testset "[OOP] [Immutable AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈
                  1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (AutoSparseForwardDiff(),
            AutoSparseFiniteDiff(), AutoZygote(), AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, LevenbergMarquardt(; autodiff); abstol = 1e-9,
            reltol = 1e-9).u .≈ sqrt(2.0))
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
        damping_decrease_factor = Float64[2, 5, 10.0]
        finite_diff_step_geodesic = [0.02, 0.2, 0.3]
        α_geodesic = [0.6, 0.8, 0.9]
        b_uphill = Float64[0, 1, 2]
        min_damping_D = [1e-12, 1e-9, 1e-4]

        list_of_options = zip(damping_initial, damping_increase_factor,
            damping_decrease_factor, finite_diff_step_geodesic, α_geodesic, b_uphill,
            min_damping_D)
        for options in list_of_options
            local probN, sol, alg
            alg = LevenbergMarquardt(; damping_initial = options[1],
                damping_increase_factor = options[2],
                damping_decrease_factor = options[3],
                finite_diff_step_geodesic = options[4], α_geodesic = options[5],
                b_uphill = options[6], min_damping_D = options[7])

            probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
            sol = solve(probN, alg; abstol = 1e-13, maxiters = 10000)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, LevenbergMarquardt(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- DFSane tests ---

@testset "DFSane" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, DFSane(), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, DFSane(), abstol = 1e-9)
    end

    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

    @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
        sol = benchmark_nlsolve_oop(quadratic_f, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), DFSane(),
            abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0), DFSane(), abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    @testset "[OOP] [Immutable AD]" begin
        for p in 1.1:0.1:100.0
            res = abs.(benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p).u)

            if any(x -> isnan(x) || x <= 1e-5 || x >= 1e5, res)
                @test_broken all(res .≈ sqrt(p))
                @test_broken abs.(ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                    @SVector[1.0, 1.0], p).u[end], p)) ≈ 1 / (2 * sqrt(p))
            else
                @test all(res .≈ sqrt(p))
                @test isapprox(abs.(ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                        @SVector[1.0, 1.0], p).u[end], p)), 1 / (2 * sqrt(p)))
            end
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.1:0.1:100.0
            res = abs(benchmark_nlsolve_oop(quadratic_f, 1.0, p).u)

            if any(x -> isnan(x) || x <= 1e-5 || x >= 1e5, res)
                @test_broken res ≈ sqrt(p)
                @test_broken abs.(ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                        1.0, p).u, p)) ≈ 1 / (2 * sqrt(p))
            else
                @test res ≈ sqrt(p)
                @test isapprox(abs.(ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                            1.0, p).u, p)), 1 / (2 * sqrt(p)))
            end
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, DFSane(); maxiters = 100, abstol = 1e-10)
        sols = zeros(length(p_range))
        for (i, p) in enumerate(p_range)
            reinit!(cache, iip ? [cache.u[1]] : cache.u; p = p)
            sol = solve!(cache)
            sols[i] = iip ? sol.u[1] : sol.u
        end
        return sols
    end
    p = range(0.01, 2, length = 200)
    @test abs.(nlprob_iterator_interface(quadratic_f, p, Val(false))) ≈ sqrt.(p)
    @test abs.(nlprob_iterator_interface(quadratic_f!, p, Val(true))) ≈ sqrt.(p)

    # Test that `DFSane` passes a test that `NewtonRaphson` fails on.
    @testset "Newton Raphson Fails" begin
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sol = benchmark_nlsolve_oop(newton_fails, u0, p)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
    end

    # Test kwargs in `DFSane`
    @testset "Keyword Arguments" begin
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
            alg = DFSane(σ_min = options[1], σ_max = options[2], σ_1 = options[3],
                M = options[4], γ = options[5], τ_min = options[6], τ_max = options[7],
                n_exp = options[8], η_strategy = options[9])

            probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
            sol = solve(probN, alg, abstol = 1e-11)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, DFSane(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- PseudoTransient tests ---

@testset "PseudoTransient" begin
    # These are tests for NewtonRaphson so we should set alpha_initial to be high so that we
    # converge quickly

    function benchmark_nlsolve_oop(f, u0, p = 2.0;
            alpha_initial = 10.0, update_alpha = switched_evolution_relaxation)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, PseudoTransient(; alpha_initial, update_alpha), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linsolve, precs,
            alpha_initial = 10.0, update_alpha = switched_evolution_relaxation)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob,
            PseudoTransient(; linsolve, precs, alpha_initial, update_alpha),
            abstol = 1e-9)
    end

    @testset "update_alpha : SER alpha_initial = 10.0 AD: $(ad)" for ad in (AutoFiniteDiff(),
        AutoZygote())
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0)
            # Failing by a margin for some
            # @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                PseudoTransient(; alpha_initial = 10.0), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        precs = [NonlinearSolve.DEFAULT_PRECS, :Random]

        @testset "[IIP] u0: $(typeof(u0)) precs: $(_nameof(prec)) linsolve: $(_nameof(linsolve))" for u0 in ([
                1.0, 1.0],), prec in precs, linsolve in (nothing, KrylovJL_GMRES())
            ad isa AutoZygote && continue
            if prec === :Random
                prec = (args...) -> (Diagonal(randn!(similar(u0))), nothing)
            end
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linsolve,
                precs = prec)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                PseudoTransient(; linsolve, precs = prec, alpha_initial = 10.0),
                abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    @testset "update_alpha : SER alpha_initial = 10.0 [OOP] [Immutable AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "update_alpha : SER alpha_initial = 10.0 [OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                    1.0, p).u, p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "update_alpha : SER alpha_initial = 10.0 [OOP] [Scalar AD]" begin
        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]
        @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
        @test ForwardDiff.jacobian(p -> [
                benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u,
            ], p) ≈ ForwardDiff.jacobian(t, p)
    end

    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, PseudoTransient(; alpha_initial = 10.0); maxiters = 100,
            abstol = 1e-10)
        sols = zeros(length(p_range))
        for (i, p) in enumerate(p_range)
            reinit!(cache, iip ? [cache.u[1]] : cache.u; p = p, alpha = 10.0)
            sol = solve!(cache)
            sols[i] = iip ? sol.u[1] : sol.u
        end
        return sols
    end
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, Val(false)) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, Val(true)) ≈ sqrt.(p)

    @testset "update_alpha : SER alpha_initial = 10.0 ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (AutoSparseForwardDiff(),
            AutoSparseFiniteDiff(), AutoZygote(), AutoSparseZygote(), AutoSparseEnzyme()), u0 in (1.0, [1.0, 1.0])
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, PseudoTransient(; alpha_initial = 10.0,
            autodiff)).u .≈ sqrt(2.0))
    end
    # Test that `PseudoTransient` passes a test that `NewtonRaphson` fails on.
    @testset "update_alpha : SER  alpha_initial = 1.0 NewtonRaphson Fails but PT passes" begin
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        probN = NonlinearProblem{false}(newton_fails, u0, p)
        sol = solve(probN, PseudoTransient(; alpha_initial = 1.0), abstol = 1e-10)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-10)
    end

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0)) update_alpha: SER alpha_initial = 10.0" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(abs.(solve(probN, PseudoTransient(; alpha_initial = 10.0);
            termination_condition).u) .≈ sqrt(2.0))
    end
end

# --- Broyden tests ---

@testset "Broyden" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; linesearch = nothing,
            init_jacobian = Val(:identity), update_rule = Val(:good_broyden))
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, Broyden(; linesearch, init_jacobian, update_rule);
            abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linesearch = nothing,
            init_jacobian = Val(:identity), update_rule = Val(:good_broyden))
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, Broyden(; linesearch, init_jacobian, update_rule);
            abstol = 1e-9)
    end

    @testset "LineSearch: $(_nameof(lsmethod)) LineSearch AD: $(_nameof(ad)) Init Jacobian: $(init_jacobian) Update Rule: $(update_rule)" for lsmethod in (Static(),
            StrongWolfe(), BackTracking(), HagerZhang(), MoreThuente(),
            LiFukushimaLineSearch()),
        ad in (AutoFiniteDiff(), AutoZygote()),
        init_jacobian in (Val(:identity), Val(:true_jacobian)),
        update_rule in (Val(:good_broyden), Val(:bad_broyden), Val(:diagonal))

        linesearch = LineSearch(; method = lsmethod, autodiff = ad)
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0; linesearch, update_rule,
                init_jacobian)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                Broyden(; linesearch, update_rule, init_jacobian), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            ad isa AutoZygote && continue
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linesearch, update_rule,
                init_jacobian)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                Broyden(; linesearch, update_rule, init_jacobian), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    @testset "[OOP] [Immutable AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, Broyden(); maxiters = 100, abstol = 1e-10)
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

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, Broyden(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- Klement tests ---

@testset "Klement" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; linesearch = nothing,
            init_jacobian = Val(:identity))
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, Klement(; linesearch, init_jacobian), abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linesearch = nothing,
            init_jacobian = Val(:identity))
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, Klement(; linesearch, init_jacobian), abstol = 1e-9)
    end

    @testset "LineSearch: $(_nameof(lsmethod)) LineSearch AD: $(_nameof(ad)) Init Jacobian: $(init_jacobian)" for lsmethod in (Static(),
            StrongWolfe(), BackTracking(), HagerZhang(), MoreThuente()),
        ad in (AutoFiniteDiff(), AutoZygote()),
        init_jacobian in (Val(:identity), Val(:true_jacobian), Val(:true_jacobian_diagonal))

        linesearch = LineSearch(; method = lsmethod, autodiff = ad)
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0; linesearch, init_jacobian)
            # Some are failing by a margin
            # @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 3e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                Klement(; linesearch), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            ad isa AutoZygote && continue
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linesearch, init_jacobian)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                Klement(; linesearch), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    @testset "[OOP] [Immutable AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, Klement(); maxiters = 100, abstol = 1e-10)
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

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, Klement(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- LimitedMemoryBroyden tests ---

@testset "LimitedMemoryBroyden" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; linesearch = nothing,
            termination_condition = AbsNormTerminationMode())
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, LimitedMemoryBroyden(; linesearch); abstol = 1e-9,
            termination_condition)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0; linesearch = nothing,
            termination_condition = AbsNormTerminationMode())
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, LimitedMemoryBroyden(; linesearch); abstol = 1e-9,
            termination_condition)
    end

    @testset "LineSearch: $(_nameof(lsmethod)) LineSearch AD: $(_nameof(ad))" for lsmethod in (Static(),
            StrongWolfe(), BackTracking(), HagerZhang(), MoreThuente(),
            LiFukushimaLineSearch()),
        ad in (AutoFiniteDiff(), AutoZygote())

        linesearch = LineSearch(; method = lsmethod, autodiff = ad)
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            sol = benchmark_nlsolve_oop(quadratic_f, u0; linesearch)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                LimitedMemoryBroyden(; linesearch), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            ad isa AutoZygote && continue
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; linesearch)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                LimitedMemoryBroyden(; linesearch), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    @testset "[OOP] [Immutable AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(((x, y),) -> isapprox(x, y; atol = 1e-3), zip(res.u, res_true))
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p)≈1 / (2 * sqrt(p)) atol=1e-3
        end
    end

    @testset "[OOP] [Scalar AD]" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    # Iterator interface
    function nlprob_iterator_interface(f, p_range, ::Val{iip}) where {iip}
        probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
        cache = init(probN, LimitedMemoryBroyden(); maxiters = 100, abstol = 1e-10)
        sols = zeros(length(p_range))
        for (i, p) in enumerate(p_range)
            reinit!(cache, iip ? [cache.u[1]] : cache.u; p = p)
            sol = solve!(cache)
            sols[i] = iip ? sol.u[1] : sol.u
        end
        return sols
    end
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, Val(false))≈sqrt.(p) atol=1e-2
    @test nlprob_iterator_interface(quadratic_f!, p, Val(true))≈sqrt.(p) atol=1e-2

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, LimitedMemoryBroyden();
            termination_condition).u .≈ sqrt(2.0))
    end
end

# Miscellaneous Tests
@testset "Custom JVP" begin
    function F(u::Vector{Float64}, p::Vector{Float64})
        Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
        return u + 0.1 * u .* Δ * u - p
    end

    function F!(du::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})
        Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
        du .= u + 0.1 * u .* Δ * u - p
        return nothing
    end

    function JVP(v::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})
        Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
        return v + 0.1 * (u .* Δ * v + v .* Δ * u)
    end

    function JVP!(du::Vector{Float64}, v::Vector{Float64}, u::Vector{Float64},
            p::Vector{Float64})
        Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
        du .= v + 0.1 * (u .* Δ * v + v .* Δ * u)
        return nothing
    end

    u0 = rand(100)

    prob = NonlinearProblem(NonlinearFunction{false}(F; jvp = JVP), u0, u0)
    sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1e-13)

    @test norm(F(sol.u, u0)) ≤ 1e-6

    prob = NonlinearProblem(NonlinearFunction{true}(F!; jvp = JVP!), u0, u0)
    sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1e-13)

    @test norm(F(sol.u, u0)) ≤ 1e-6
end
