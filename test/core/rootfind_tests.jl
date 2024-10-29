@testsetup module CoreRootfindTesting
using Reexport
@reexport using BenchmarkTools, LinearSolve, NonlinearSolve, StaticArrays, Random,
                LinearAlgebra, ForwardDiff, Zygote, Enzyme, SparseConnectivityTracer,
                NonlinearSolveBase
using LineSearches: LineSearches

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
               216.40425613334457 ./ (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^
            2.0) .- 0.0011552453009332421u .- p
end

const TERMINATION_CONDITIONS = [
    NormTerminationMode(Base.Fix1(maximum, abs)),
    RelTerminationMode(),
    RelNormTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
    AbsTerminationMode(),
    AbsNormTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs))
]

function benchmark_nlsolve_oop(f, u0, p = 2.0; solver, kwargs...)
    prob = NonlinearProblem{false}(f, u0, p)
    return solve(prob, solver; abstol = 1e-9, kwargs...)
end

function benchmark_nlsolve_iip(f, u0, p = 2.0; solver, kwargs...)
    prob = NonlinearProblem{true}(f, u0, p)
    return solve(prob, solver; abstol = 1e-9, kwargs...)
end

function nlprob_iterator_interface(f, p_range, ::Val{iip}, solver) where {iip}
    probN = NonlinearProblem{iip}(f, iip ? [0.5] : 0.5, p_range[begin])
    cache = init(probN, solver; maxiters = 100, abstol = 1e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, iip ? [cache.u[1]] : cache.u; p = p)
        sol = solve!(cache)
        sols[i] = iip ? sol.u[1] : sol.u
    end
    return sols
end

for alg in (:Static, :StrongWolfe, :BackTracking, :MoreThuente, :HagerZhang)
    algname = Symbol(:LineSearches, alg)
    @eval function $(algname)(args...; autodiff = nothing, initial_alpha = true, kwargs...)
        return LineSearch.LineSearchesJL(;
            method = LineSearches.$(alg)(args...; kwargs...), autodiff, initial_alpha)
    end
end

export nlprob_iterator_interface, benchmark_nlsolve_oop, benchmark_nlsolve_iip,
       TERMINATION_CONDITIONS, _nameof, newton_fails, quadratic_f, quadratic_f!
export LineSearchesStatic, LineSearchesStrongWolfe, LineSearchesBackTracking,
       LineSearchesMoreThuente, LineSearchesHagerZhang

end

# --- NewtonRaphson tests ---

@testitem "NewtonRaphson" setup=[CoreRootfindTesting] tags=[:core] begin
    @testset "LineSearch: $(_nameof(linesearch)) LineSearch AD: $(_nameof(ad))" for ad in (
            AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()
        ),
        linesearch in (
            LineSearchesStatic(; autodiff = ad), LineSearchesStrongWolfe(; autodiff = ad),
            LineSearchesBackTracking(; autodiff = ad), BackTracking(; autodiff = ad),
            LineSearchesHagerZhang(; autodiff = ad),
            LineSearchesMoreThuente(; autodiff = ad)
        )

        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            solver = NewtonRaphson(; linesearch)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                NewtonRaphson(), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        precs = [(u0) -> nothing,
            u0 -> ((args...) -> (Diagonal(rand!(similar(u0))), nothing))]

        @testset "[IIP] u0: $(typeof(u0)) precs: $(_nameof(prec)) linsolve: $(_nameof(linsolve))" for u0 in ([
                1.0, 1.0],),
            prec in precs,
            linsolve in (nothing, KrylovJL_GMRES(), \)

            ad isa AutoZygote && continue
            if prec === :Random
                prec = (args...) -> (Diagonal(randn!(similar(u0))), nothing)
            end
            solver = NewtonRaphson(; linsolve, precs = prec(u0), linesearch)
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                NewtonRaphson(; linsolve, precs = prec(u0)), abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    # Iterator interface
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, Val(false), NewtonRaphson()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, Val(true), NewtonRaphson()) ≈ sqrt.(p)

    @testset "Sparsity ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (
            AutoForwardDiff(), AutoFiniteDiff(), AutoZygote(), AutoEnzyme()),
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(
            NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0)
        @test all(solve(probN, NewtonRaphson(; autodiff)).u .≈ sqrt(2.0))
    end

    @testset "Termination condition: $(_nameof(termination_condition)) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, NewtonRaphson(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- TrustRegion tests ---

@testitem "TrustRegion" setup=[CoreRootfindTesting] tags=[:core] begin
    radius_update_schemes = [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
    linear_solvers = [nothing, LUFactorization(), KrylovJL_GMRES(), \]

    @testset "[OOP] u0: $(typeof(u0)) $(radius_update_scheme) $(_nameof(linsolve))" for u0 in u0s,
        radius_update_scheme in radius_update_schemes,
        linsolve in linear_solvers

        abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)

        solver = TrustRegion(; radius_update_scheme, linsolve)
        sol = benchmark_nlsolve_oop(quadratic_f, u0; solver, abstol)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< abstol)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            TrustRegion(; radius_update_scheme, linsolve); abstol)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0)) $(radius_update_scheme) $(_nameof(linsolve))" for u0 in ([
            1.0, 1.0],),
        radius_update_scheme in radius_update_schemes,
        linsolve in linear_solvers

        abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)
        solver = TrustRegion(; radius_update_scheme, linsolve)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver, abstol)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< abstol)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
            TrustRegion(; radius_update_scheme); abstol)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    # Iterator interface
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, Val(false), TrustRegion()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, Val(true), TrustRegion()) ≈ sqrt.(p)

    @testset "$(_nameof(autodiff)) u0: $(_nameof(u0)) $(radius_update_scheme)" for autodiff in (
            AutoForwardDiff(), AutoFiniteDiff(), AutoZygote(), AutoEnzyme()),
        u0 in (1.0, [1.0, 1.0]),
        radius_update_scheme in radius_update_schemes

        probN = NonlinearProblem(
            NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0)
        @test all(solve(probN, TrustRegion(; autodiff, radius_update_scheme)).u .≈
                  sqrt(2.0))
    end

    # Test that `TrustRegion` passes a test that `NewtonRaphson` fails on.
    @testset "Newton Raphson Fails: radius_update_scheme: $(radius_update_scheme)" for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        solver = TrustRegion(; radius_update_scheme)
        sol = benchmark_nlsolve_oop(newton_fails, u0, p; solver)
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

        list_of_options = zip(
            max_trust_radius, initial_trust_radius, step_threshold, shrink_threshold,
            expand_threshold, shrink_factor, expand_factor, max_shrink_times)
        for options in list_of_options
            local probN, sol, alg
            alg = TrustRegion(
                max_trust_radius = options[1], initial_trust_radius = options[2],
                step_threshold = options[3], shrink_threshold = options[4],
                expand_threshold = options[5], shrink_factor = options[6],
                expand_factor = options[7], max_shrink_times = options[8])

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

            solver = TrustRegion(; radius_update_scheme)
            sol_iip = benchmark_nlsolve_iip(quadratic_f!, u0; solver, maxiters)
            sol_oop = benchmark_nlsolve_oop(quadratic_f, u0; solver, maxiters)
            @test sol_iip.u ≈ sol_oop.u
        end
    end

    @testset "Termination condition: $(_nameof(termination_condition)) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, TrustRegion(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- LevenbergMarquardt tests ---

@testitem "LevenbergMarquardt" setup=[CoreRootfindTesting] tags=[:core] begin
    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
    @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
        sol = benchmark_nlsolve_oop(quadratic_f, u0; solver = LevenbergMarquardt())
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
            LevenbergMarquardt(), abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver = LevenbergMarquardt())
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0),
            LevenbergMarquardt(), abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (
            AutoForwardDiff(), AutoFiniteDiff(), AutoZygote(), AutoEnzyme()),
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(
            NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0)
        @test all(solve(
            probN, LevenbergMarquardt(; autodiff); abstol = 1e-9, reltol = 1e-9).u .≈
                  sqrt(2.0))
    end

    # Test that `LevenbergMarquardt` passes a test that `NewtonRaphson` fails on.
    @testset "Newton Raphson Fails" begin
        u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sol = benchmark_nlsolve_oop(newton_fails, u0, p; solver = LevenbergMarquardt())
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
    end

    # Iterator interface
    p = range(0.01, 2, length = 200)
    @test abs.(nlprob_iterator_interface(
        quadratic_f, p, Val(false), LevenbergMarquardt())) ≈ sqrt.(p)
    @test abs.(nlprob_iterator_interface(
        quadratic_f!, p, Val(true), LevenbergMarquardt())) ≈ sqrt.(p)

    # Test kwargs in `LevenbergMarquardt`
    @testset "Keyword Arguments" begin
        damping_initial = [0.5, 2.0, 5.0]
        damping_increase_factor = [1.5, 3.0, 10.0]
        damping_decrease_factor = Float64[2, 5, 10.0]
        finite_diff_step_geodesic = [0.02, 0.2, 0.3]
        α_geodesic = [0.6, 0.8, 0.9]
        b_uphill = Float64[0, 1, 2]
        min_damping_D = [1e-12, 1e-9, 1e-4]

        list_of_options = zip(
            damping_initial, damping_increase_factor, damping_decrease_factor,
            finite_diff_step_geodesic, α_geodesic, b_uphill, min_damping_D)
        for options in list_of_options
            local probN, sol, alg
            alg = LevenbergMarquardt(;
                damping_initial = options[1], damping_increase_factor = options[2],
                damping_decrease_factor = options[3],
                finite_diff_step_geodesic = options[4], α_geodesic = options[5],
                b_uphill = options[6], min_damping_D = options[7])

            probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
            sol = solve(probN, alg; abstol = 1e-13, maxiters = 10000)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end

    @testset "Termination condition: $(_nameof(termination_condition)) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, LevenbergMarquardt(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- PseudoTransient tests ---

@testitem "PseudoTransient" setup=[CoreRootfindTesting] tags=[:core] begin
    # These are tests for NewtonRaphson so we should set alpha_initial to be high so that we
    # converge quickly
    @testset "PT: alpha_initial = 10.0 PT AD: $(ad)" for ad in (
        AutoFiniteDiff(), AutoZygote())
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            solver = PseudoTransient(; alpha_initial = 10.0)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; solver)
            # Failing by a margin for some
            # @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0),
                PseudoTransient(alpha_initial = 10.0), abstol = 1e-9)
            @test (@ballocated solve!($cache)) < 200
        end

        precs = [nothing, :Random]

        @testset "[IIP] u0: $(typeof(u0)) precs: $(_nameof(prec)) linsolve: $(_nameof(linsolve))" for u0 in ([
                1.0, 1.0],),
            prec in precs,
            linsolve in (nothing, KrylovJL_GMRES(), \)

            ad isa AutoZygote && continue
            if prec === :Random
                prec = (args...) -> (Diagonal(randn!(similar(u0))), nothing)
            end
            solver = PseudoTransient(; alpha_initial = 10.0, linsolve, precs = prec)
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)

            cache = init(
                NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver; abstol = 1e-9)
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end

    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(
        quadratic_f, p, Val(false), PseudoTransient(; alpha_initial = 10.0)) ≈ sqrt.(p)
    @test nlprob_iterator_interface(
        quadratic_f!, p, Val(true), PseudoTransient(; alpha_initial = 10.0)) ≈ sqrt.(p)

    @testset "ADType: $(autodiff) u0: $(_nameof(u0))" for autodiff in (
            AutoForwardDiff(), AutoFiniteDiff(), AutoZygote(), AutoEnzyme()),
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(
            NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0)
        @test all(solve(probN, PseudoTransient(; alpha_initial = 10.0, autodiff)).u .≈
                  sqrt(2.0))
    end

    @testset "Termination condition: $(_nameof(termination_condition)) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(
            probN, PseudoTransient(; alpha_initial = 10.0); termination_condition).u .≈
                  sqrt(2.0))
    end
end

# Miscellaneous Tests
@testitem "Custom JVP" setup=[CoreRootfindTesting] tags=[:core] begin
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

    function JVP!(
            du::Vector{Float64}, v::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})
        Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
        du .= v + 0.1 * (u .* Δ * v + v .* Δ * u)
        return nothing
    end

    u0 = rand(100)

    prob = NonlinearProblem(NonlinearFunction{false}(F; jvp = JVP), u0, u0)
    sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1e-13)
    @test norm(sol.resid, Inf) ≤ 1e-6
    sol = solve(
        prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
        abstol = 1e-13)
    @test norm(sol.resid, Inf) ≤ 1e-6

    prob = NonlinearProblem(NonlinearFunction{true}(F!; jvp = JVP!), u0, u0)
    sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1e-13)
    @test norm(sol.resid, Inf) ≤ 1e-6
    sol = solve(
        prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
        abstol = 1e-13)
    @test norm(sol.resid, Inf) ≤ 1e-6
end
