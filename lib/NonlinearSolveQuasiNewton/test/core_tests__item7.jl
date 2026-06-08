using NonlinearSolveQuasiNewton
include("setup_corerootfindtesting.jl")

using ADTypes, LineSearch
using LineSearches: LineSearches
using BenchmarkTools: @ballocated
using StaticArrays: @SVector
using Zygote, ForwardDiff, FiniteDiff

# Conditionally import Enzyme only if not on Julia prerelease
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    using Enzyme
end

# Filter autodiff backends based on Julia version
autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
if isempty(VERSION.prerelease) && VERSION < v"1.12"
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
            LiFukushimaLineSearch(),
        )
        @testset "[OOP] u0: $(typeof(u0))" for u0 in (ones(32), @SVector(ones(2)), 1.0)
            broken = Sys.iswindows() && u0 isa Vector{Float64} &&
                linesearch isa BackTracking && ad isa AutoFiniteDiff

            solver = LimitedMemoryBroyden(; linesearch)
            sol = solve_oop(quadratic_f, u0; solver)
            @test SciMLBase.successful_retcode(sol) broken = broken
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1.0e-9 broken = broken

            cache = init(
                NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1.0e-9
            )
            @test (@ballocated solve!($cache)) ≤ 400
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in (ones(32),)
            ad isa AutoZygote && continue

            broken = Sys.iswindows() && u0 isa Vector{Float64} &&
                linesearch isa BackTracking && ad isa AutoFiniteDiff

            solver = LimitedMemoryBroyden(; linesearch)
            sol = solve_iip(quadratic_f!, u0; solver)
            @test SciMLBase.successful_retcode(sol) broken = broken
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1.0e-9 broken = broken

            cache = init(
                NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol = 1.0e-9
            )
            @test (@ballocated solve!($cache)) ≤ 64
        end
    end
end
