using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using StaticArrays: @SVector

@testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in
    TERMINATION_CONDITIONS

    @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        sol = solve(probN, LevenbergMarquardt(); termination_condition)
        err = maximum(abs, quadratic_f(sol.u, 2.0))
        @test err < 1.0e-9
    end
end
