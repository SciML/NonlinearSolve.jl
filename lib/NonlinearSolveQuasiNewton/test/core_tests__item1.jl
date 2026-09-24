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

u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

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
        @testset for init_jacobian in (Val(:identity), Val(:true_jacobian)),
                update_rule in (Val(:good_broyden), Val(:bad_broyden), Val(:diagonal))

            @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                    [1.0, 1.0], @SVector[1.0, 1.0], 1.0,
                )
                solver = Broyden(; linesearch, init_jacobian, update_rule)
                sol = solve_oop(quadratic_f, u0; solver)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1.0e-9

                cache = init(
                    NonlinearProblem{false}(quadratic_f, u0, 2.0), solver, abstol = 1.0e-9
                )
                @test (@ballocated solve!($cache)) < 200
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                ad isa AutoZygote && continue

                solver = Broyden(; linesearch, init_jacobian, update_rule)
                sol = solve_iip(quadratic_f!, u0; solver)
                @test SciMLBase.successful_retcode(sol)
                err = maximum(abs, quadratic_f(sol.u, 2.0))
                @test err < 1.0e-9

                cache = init(
                    NonlinearProblem{true}(quadratic_f!, u0, 2.0), solver, abstol = 1.0e-9
                )
                @test (@ballocated solve!($cache)) ≤ 64
            end
        end
    end
end
