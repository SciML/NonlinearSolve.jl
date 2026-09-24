using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using LinearAlgebra, Random, LinearSolve
using BenchmarkTools: @ballocated
using StaticArrays: @SVector

@testset for (concrete_jac, linsolve) in (
        (Val(false), KrylovJL_CG(; precs = nothing)),
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
    )
    @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0])
        solver = NewtonRaphson(; forcing = EisenstatWalkerForcing2(), linsolve, concrete_jac)
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
        solver = NewtonRaphson(; forcing = EisenstatWalkerForcing2(), linsolve, concrete_jac)

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
