using NonlinearSolveSpectralMethods
include("setup_corerootfindtesting.jl")

using BenchmarkTools: @ballocated
using StaticArrays: @SVector

u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

@testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
    sol = solve_oop(quadratic_f, u0; solver = DFSane())
    @test SciMLBase.successful_retcode(sol)
    err = maximum(abs, quadratic_f(sol.u, 2.0))
    @test err < 1.0e-9

    cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), DFSane(), abstol = 1.0e-9)
    @test (@ballocated solve!($cache)) < 200
end

@testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
    sol = solve_iip(quadratic_f!, u0; solver = DFSane())
    @test SciMLBase.successful_retcode(sol)
    err = maximum(abs, quadratic_f(sol.u, 2.0))
    @test err < 1.0e-9

    cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0), DFSane(), abstol = 1.0e-9)
    @test (@ballocated solve!($cache)) ≤ 64
end
