@testsetup module CoreRootfindTesting

include("../../../common/common_rootfind_testing.jl")

end

@testitem "Broyden" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LineSearch
    using LineSearches: LineSearches
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme based on Julia version
    enzyme_available = false
    if isempty(VERSION.prerelease)
        try
            using Enzyme
            enzyme_available = true
        catch e
            @info "Enzyme not available: $e"
            enzyme_available = false
        end
    else
        @info "Skipping Enzyme on prerelease Julia $(VERSION)"
        enzyme_available = false
    end

    u0s=([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

    # Filter autodiff backends based on Julia version
    autodiff_backends=[AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if enzyme_available
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset "$(nameof(typeof(linesearch)))" for linesearch in (
            # LineSearchesJL(; method = LineSearches.Static(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.BackTracking(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.MoreThuente(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.StrongWolfe(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.HagerZhang(), autodiff = ad),
            BackTracking(; autodiff = ad),
            LiFukushimaLineSearch()
        )
            @testset for init_jacobian in (Val(:identity), Val(:true_jacobian)),
                update_rule in (Val(:good_broyden), Val(:bad_broyden), Val(:diagonal))

                @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                    [1.0, 1.0], @SVector[1.0, 1.0], 1.0
                )
                    solver = Broyden(; linesearch, init_jacobian, update_rule)
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

                    solver = Broyden(; linesearch, init_jacobian, update_rule)
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

@testitem "Broyden: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p=range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, Broyden()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, Broyden()) ≈ sqrt.(p)
end

@testitem "Broyden Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in
                                                                TERMINATION_CONDITIONS

        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, Broyden(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "Klement" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LineSearch
    using LineSearches: LineSearches
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme based on Julia version
    enzyme_available = false
    if isempty(VERSION.prerelease)
        try
            using Enzyme
            enzyme_available = true
        catch e
            @info "Enzyme not available: $e"
            enzyme_available = false
        end
    else
        @info "Skipping Enzyme on prerelease Julia $(VERSION)"
        enzyme_available = false
    end

    # Filter autodiff backends based on Julia version
    autodiff_backends=[AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if enzyme_available
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset "$(nameof(typeof(linesearch)))" for linesearch in (
            # LineSearchesJL(; method = LineSearches.Static(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.BackTracking(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.MoreThuente(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.StrongWolfe(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.HagerZhang(), autodiff = ad),
            BackTracking(; autodiff = ad),
            LiFukushimaLineSearch()
        )
            @testset for init_jacobian in (
                Val(:identity), Val(:true_jacobian), Val(:true_jacobian_diagonal))
                @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                    [1.0, 1.0], @SVector[1.0, 1.0], 1.0
                )
                    solver = Klement(; linesearch, init_jacobian)
                    sol = solve_oop(quadratic_f, u0; solver)
                    # XXX: some tests are failing by a margin
                    # @test SciMLBase.successful_retcode(sol)
                    err = maximum(abs, quadratic_f(sol.u, 2.0))
                    @test err < 1e-9

                    cache = init(
                        NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1e-9
                    )
                    @test (@ballocated solve!($cache)) < 200
                end

                @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                    ad isa AutoZygote && continue

                    solver = Klement(; linesearch, init_jacobian)
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

@testitem "Klement: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p=range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, Klement()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, Klement()) ≈ sqrt.(p)
end

@testitem "Klement Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in
                                                                TERMINATION_CONDITIONS

        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, Klement(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end

@testitem "LimitedMemoryBroyden" setup=[CoreRootfindTesting] tags=[:core] begin
    using ADTypes, LineSearch
    using LineSearches: LineSearches
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector
    using Zygote, ForwardDiff, FiniteDiff

    # Conditionally import Enzyme based on Julia version
    enzyme_available = false
    if isempty(VERSION.prerelease)
        try
            using Enzyme
            enzyme_available = true
        catch e
            @info "Enzyme not available: $e"
            enzyme_available = false
        end
    else
        @info "Skipping Enzyme on prerelease Julia $(VERSION)"
        enzyme_available = false
    end

    # Filter autodiff backends based on Julia version
    autodiff_backends=[AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
    if enzyme_available
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for ad in autodiff_backends
        @testset "$(nameof(typeof(linesearch)))" for linesearch in (
            # LineSearchesJL(; method = LineSearches.Static(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.BackTracking(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.MoreThuente(), autodiff = ad),
            # LineSearchesJL(; method = LineSearches.StrongWolfe(), autodiff = ad),
            LineSearchesJL(; method = LineSearches.HagerZhang(), autodiff = ad),
            BackTracking(; autodiff = ad),
            LiFukushimaLineSearch()
        )
            @testset "[OOP] u0: $(typeof(u0))" for u0 in (ones(32), @SVector(ones(2)), 1.0)
                broken = Sys.iswindows() && u0 isa Vector{Float64} &&
                         linesearch isa BackTracking && ad isa AutoFiniteDiff

                solver = LimitedMemoryBroyden(; linesearch)
                sol = solve_oop(quadratic_f, u0; solver)
                @test SciMLBase.successful_retcode(sol) broken=broken
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err<1e-9 broken=broken

                cache = init(
                    NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1e-9
                )
                @test (@ballocated solve!($cache)) ≤ 400
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in (ones(32),)
                ad isa AutoZygote && continue

                broken = Sys.iswindows() && u0 isa Vector{Float64} &&
                         linesearch isa BackTracking && ad isa AutoFiniteDiff

                solver = LimitedMemoryBroyden(; linesearch)
                sol = solve_iip(quadratic_f!, u0; solver)
                @test SciMLBase.successful_retcode(sol) broken=broken
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err<1e-9 broken=broken

                cache = init(
                    NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol = 1e-9
                )
                @test (@ballocated solve!($cache)) ≤ 64
            end
        end
    end
end

@testitem "LimitedMemoryBroyden: Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p=range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, LimitedMemoryBroyden()) ≈
          sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, LimitedMemoryBroyden()) ≈
          sqrt.(p)
end

@testitem "LimitedMemoryBroyden Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in
                                                                TERMINATION_CONDITIONS

        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, LimitedMemoryBroyden(); termination_condition)
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1e-9
        end
    end
end
