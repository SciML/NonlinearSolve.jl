using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using ADTypes, LinearSolve, LinearAlgebra
using BenchmarkTools: @ballocated
using StaticArrays: @SVector
using Zygote, ForwardDiff, FiniteDiff

# Conditionally import Enzyme only if not on Julia prerelease
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    using Enzyme
end

radius_update_schemes = [
    RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
    RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
    RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin,
]

# Filter autodiff backends based on Julia version
autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    push!(autodiff_backends, AutoEnzyme())
end

@testset for ad in autodiff_backends
    @testset for radius_update_scheme in radius_update_schemes,
            linsolve in (nothing, LUFactorization(), KrylovJL_GMRES(), \)

        @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], 1.0, @SVector[1.0, 1.0], 1.0,
            )
            abstol = ifelse(linsolve isa KrylovJL, 1.0e-6, 1.0e-9)
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

            abstol = ifelse(linsolve isa KrylovJL, 1.0e-6, 1.0e-9)
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
