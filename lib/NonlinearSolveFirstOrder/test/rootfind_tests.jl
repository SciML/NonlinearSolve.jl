@testsetup module CoreRootfindTesting

include("../../../common/common_rootfind_testing.jl")

end

@testitem "NewtonRaphson" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LineSearch, LinearAlgebra, Random, LinearSolve
    using LineSearches: LineSearches
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease)
        using Enzyme
    end

    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

    # Filter autodiff backends based on Julia version
    autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if isempty(VERSION.prerelease)
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset "$(nameof(typeof(linesearch)))" for linesearch in (
            LineSearchesJL(; method = LineSearches.Static(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.BackTracking(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.MoreThuente(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.StrongWolfe(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.HagerZhang(), autodiff = ad),
            BackTracking(; autodiff = ad)
        )
            @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
                solver = NewtonRaphson(; linesearch, autodiff = ad)
                sol = solve_oop(quadratic_f, u0; solver)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1e-9

                cache = init(
                    NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1e-9
                )
                @test (@ballocated solve!($cache)) < 200
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                ad isa AutoZygote && continue

                @testset for (concrete_jac, linsolve) in (
                    (Val(false), nothing),
                    (Val(false), KrylovJL_GMRES(; precs = nothing)),
                    (
                        Val(true),
                        KrylovJL_GMRES(;
                            precs = (A,
                            p = nothing) -> (
                                Diagonal(randn!(similar(A, size(A, 1)))), LinearAlgebra.I
                            )
                        )
                    ),
                    (Val(false), \)
                )
                    solver = NewtonRaphson(;
                        linsolve, linesearch, autodiff = ad, concrete_jac
                    )

                    sol = solve_iip(quadratic_f!, u0; solver)
                    @test SciMLBase.successful_retcode(sol)
                    err = maximum(abs, quadratic_f(sol.u, 2.0))
                    @test err < 1e-9

                    cache = init(
                        NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol = 1e-9
                    )
                    @test (@ballocated solve!($cache)) ≤ 64
                end
            end
        end
    end
end

@testitem "NewtonRaphson: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, NewtonRaphson()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, NewtonRaphson()) ≈ sqrt.(p)
end

@testitem "NewtonRaphson Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in TERMINATION_CONDITIONS
        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, NewtonRaphson(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "PseudoTransient" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, Random, LinearSolve, LinearAlgebra
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease)
        using Enzyme
    end

    preconditioners = [
        (u0) -> nothing,
        u0 -> ((args...) -> (Diagonal(rand!(similar(u0))), nothing))
    ]

    # Filter autodiff backends based on Julia version
    autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if isempty(VERSION.prerelease)
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
            solver = PseudoTransient(; alpha_initial = 10.0, autodiff = ad)
            sol = solve_oop(quadratic_f, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9

            cache = init(
                NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1e-9
            )
            @test (@ballocated solve!($cache)) < 200
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            ad isa AutoZygote && continue

            @testset for (concrete_jac, linsolve) in (
                (Val(false), nothing),
                (Val(false), KrylovJL_GMRES(; precs = nothing)),
                (
                    Val(true),
                    KrylovJL_GMRES(;
                        precs = (A,
                        p = nothing) -> (
                            Diagonal(randn!(similar(A, size(A, 1)))), LinearAlgebra.I
                        )
                    )
                ),
                (Val(false), \)
            )
                solver = PseudoTransient(;
                    alpha_initial = 10.0, linsolve, autodiff = ad, concrete_jac
                )
                sol = solve_iip(quadratic_f!, u0; solver)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1e-9

                cache = init(
                    NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol = 1e-9
                )
                @test (@ballocated solve!($cache)) ≤ 64
            end
        end
    end
end

@testitem "PseudoTransient: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(
        quadratic_f, p, false, PseudoTransient(; alpha_initial = 10.0)
    ) ≈ sqrt.(p)
    @test nlprob_iterator_interface(
        quadratic_f!, p, true, PseudoTransient(; alpha_initial = 10.0)
    ) ≈ sqrt.(p)
end

@testitem "PseudoTransient Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in TERMINATION_CONDITIONS
        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, PseudoTransient(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "TrustRegion" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LinearSolve, LinearAlgebra
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease)
        using Enzyme
    end

    radius_update_schemes = [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin
    ]

    # Filter autodiff backends based on Julia version
    autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if isempty(VERSION.prerelease)
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset for radius_update_scheme in radius_update_schemes,
            linsolve in (nothing, LUFactorization(), KrylovJL_GMRES(), \)

            @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], 1.0, @SVector[1.0, 1.0], 1.0)
                abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)
                solver = TrustRegion(; radius_update_scheme, linsolve)
                sol = solve_oop(quadratic_f, u0; solver, abstol)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < abstol

                cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol)
                @test (@ballocated solve!($cache)) < 200
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                ad isa AutoZygote && continue

                abstol = ifelse(linsolve isa KrylovJL, 1e-6, 1e-9)
                solver = TrustRegion(; radius_update_scheme, linsolve)
                sol = solve_iip(quadratic_f!, u0; solver, abstol)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < abstol

                cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol)
                @test (@ballocated solve!($cache)) ≤ 64
            end
        end
    end
end

@testitem "TrustRegion: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, TrustRegion()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, TrustRegion()) ≈ sqrt.(p)
end

@testitem "TrustRegion NewtonRaphson Fails" setup=[CoreRootfindTesting] tags=[:core] begin
    u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
    p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_oop(newton_fails, u0, p; solver = TrustRegion())
    @test SciMLBase.successful_retcode(sol)
    @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
end

@testitem "TrustRegion: Kwargs" setup=[CoreRootfindTesting] tags=[:core] begin
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
        expand_threshold, shrink_factor, expand_factor, max_shrink_times
    )

    for options in list_of_options
        alg = TrustRegion(;
            max_trust_radius = options[1], initial_trust_radius = options[2],
            step_threshold = options[3], shrink_threshold = options[4],
            expand_threshold = options[5], shrink_factor = options[6],
            expand_factor = options[7], max_shrink_times = options[8]
        )

        sol = solve_oop(quadratic_f, [1.0, 1.0], 2.0; solver = alg)
        @test SciMLBase.successful_retcode(sol)
        err = maximum(abs, quadratic_f(sol.u, 2.0))
        @test err < 1e-9
    end
end

@testitem "TrustRegion OOP / IIP Consistency" setup=[CoreRootfindTesting] tags=[:core] begin
    maxiterations = [2, 3, 4, 5]
    u0 = [1.0, 1.0]
    @testset for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin
    ]
        @testset for maxiters in maxiterations
            solver = TrustRegion(; radius_update_scheme)
            sol_iip = solve_iip(quadratic_f!, u0; solver, maxiters)
            sol_oop = solve_oop(quadratic_f, u0; solver, maxiters)
            @test sol_iip.u ≈ sol_oop.u
        end
    end
end

@testitem "TrustRegion Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in TERMINATION_CONDITIONS
        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, TrustRegion(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "LevenbergMarquardt" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LinearSolve, LinearAlgebra
    using BenchmarkTools: @ballocated
    using StaticArrays: SVector, @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease)
        using Enzyme
    end

    # Filter autodiff backends based on Julia version
    autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if isempty(VERSION.prerelease)
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        solver = LevenbergMarquardt(; autodiff = ad)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            if ad isa ADTypes.AutoZygote && u0 isa SVector
                # Zygote converts SVector to a Matrix that triggers a bug upstream
                @test_broken solve_oop(quadratic_f, u0; solver)
                continue
            end

            sol = solve_oop(quadratic_f, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9

            cache = init(
                NonlinearProblem{false}(quadratic_f, u0, 2.0),
                LevenbergMarquardt(), abstol = 1e-9
            )
            @test (@ballocated solve!($cache)) < 200
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            ad isa AutoZygote && continue

            sol = solve_iip(quadratic_f!, u0; solver)
            @test SciMLBase.successful_retcode(sol)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9

            cache = init(
                NonlinearProblem{true}(quadratic_f!, u0, 2.0),
                LevenbergMarquardt(), abstol = 1e-9
            )
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end
end

@testitem "LevenbergMarquardt NewtonRaphson Fails" setup=[CoreRootfindTesting] tags=[:core] begin
    u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
    p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_oop(newton_fails, u0, p; solver = LevenbergMarquardt())
    @test SciMLBase.successful_retcode(sol)
    @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
end

@testitem "LevenbergMarquardt: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, LevenbergMarquardt()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, LevenbergMarquardt()) ≈ sqrt.(p)
end

@testitem "LevenbergMarquardt Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in TERMINATION_CONDITIONS
        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, LevenbergMarquardt(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "LevenbergMarquardt: Kwargs" setup=[CoreRootfindTesting] tags=[:core] begin
    damping_initial = [0.5, 2.0, 5.0]
    damping_increase_factor = [1.5, 3.0, 10.0]
    damping_decrease_factor = Float64[2, 5, 10.0]
    finite_diff_step_geodesic = [0.02, 0.2, 0.3]
    α_geodesic = [0.6, 0.8, 0.9]
    b_uphill = Float64[0, 1, 2]
    min_damping_D = [1e-12, 1e-9, 1e-4]

    list_of_options = zip(
        damping_initial, damping_increase_factor, damping_decrease_factor,
        finite_diff_step_geodesic, α_geodesic, b_uphill, min_damping_D
    )
    for options in list_of_options
        alg = LevenbergMarquardt(;
            damping_initial = options[1], damping_increase_factor = options[2],
            damping_decrease_factor = options[3],
            finite_diff_step_geodesic = options[4], α_geodesic = options[5],
            b_uphill = options[6], min_damping_D = options[7]
        )

        sol = solve_oop(quadratic_f, [1.0, 1.0], 2.0; solver = alg, maxiters = 10000)
        @test SciMLBase.successful_retcode(sol)
        err = maximum(abs, quadratic_f(sol.u, 2.0))
        @test err < 1e-9
    end
end

@testitem "Simple Sparse AutoDiff" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, SparseConnectivityTracer, SparseMatrixColorings

    # Filter autodiff backends based on Julia version
    autodiff_backends = [AutoForwardDiff(), AutoFiniteDiff(), AutoZygote()]
    if isempty(VERSION.prerelease)
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset for u0 in ([1.0, 1.0], 1.0)
            prob = NonlinearProblem(
                NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0
            )

            @testset "Newton Raphson" begin
                sol = solve(prob, NewtonRaphson(; autodiff = ad))
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1e-9
            end

            @testset "Trust Region" begin
                sol = solve(prob, TrustRegion(; autodiff = ad))
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1e-9
            end
        end
    end
end

@testitem "Custom JVP" setup=[CoreRootfindTesting] tags=[:core] begin
    using LinearAlgebra, LinearSolve, ADTypes

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
    err = maximum(abs, sol.resid)
    @test err < 1e-6

    sol = solve(
        prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
        abstol = 1e-13
    )
    err = maximum(abs, sol.resid)
    @test err < 1e-6

    prob = NonlinearProblem(NonlinearFunction{true}(F!; jvp = JVP!), u0, u0)
    sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1e-13)
    err = maximum(abs, sol.resid)
    @test err < 1e-6

    sol = solve(
        prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
        abstol = 1e-13
    )
    err = maximum(abs, sol.resid)
    @test err < 1e-6
end
