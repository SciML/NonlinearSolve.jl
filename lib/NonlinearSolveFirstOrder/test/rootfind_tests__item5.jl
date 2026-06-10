using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using ADTypes, Random, LinearSolve, LinearAlgebra
using BenchmarkTools: @ballocated
using StaticArrays: @SVector
using Zygote, ForwardDiff, FiniteDiff

# Conditionally import Enzyme only if not on Julia prerelease
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    using Enzyme
end

preconditioners = [
    (u0) -> nothing,
    u0 -> ((args...) -> (Diagonal(rand!(similar(u0))), nothing)),
]

# Filter autodiff backends based on Julia version
autodiff_backends = [AutoForwardDiff(), AutoZygote(), AutoFiniteDiff()]
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    push!(autodiff_backends, AutoEnzyme())
end

@testset for ad in autodiff_backends
    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

    @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
        solver = PseudoTransient(; alpha_initial = 10.0, autodiff = ad)
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

        @testset for (concrete_jac, linsolve) in (
                (Val(false), nothing),
                (Val(false), KrylovJL_GMRES(; precs = nothing)),
                (
                    Val(true),
                    KrylovJL_GMRES(;
                        precs = (
                            A,
                            p = nothing,
                        ) -> (
                            Diagonal(randn!(similar(A, size(A, 1)))), LinearAlgebra.I,
                        )
                    ),
                ),
                (Val(false), \),
            )
            solver = PseudoTransient(;
                alpha_initial = 10.0, linsolve, autodiff = ad, concrete_jac
            )
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
