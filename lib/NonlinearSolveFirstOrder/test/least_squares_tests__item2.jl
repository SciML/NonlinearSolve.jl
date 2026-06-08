using NonlinearSolveFirstOrder

using NonlinearSolveFirstOrder, LinearAlgebra, StaticArrays

# Overdetermined: 4 residuals, 2 unknowns (issue #560)
function f_static(u, p)
    x, y = u
    return SA[
        sin(x) + y^2 - 1.0,
        x^2 + y - 2.0,
        exp(x) + x * y - 3.0,
        x + y - 1.0,
    ]
end
f_vec(u, p) = collect(f_static(u, p))

prob_static = NonlinearLeastSquaresProblem(NonlinearFunction(f_static), SA[0.0, 0.0])
prob_vec = NonlinearLeastSquaresProblem(NonlinearFunction(f_vec), [0.0, 0.0])

for solver in (GaussNewton(), LevenbergMarquardt(), TrustRegion())
    sol_static = solve(prob_static, solver; maxiters = 10000, abstol = 1.0e-6)
    sol_vec = solve(prob_vec, solver; maxiters = 10000, abstol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol_static)

    # We don't test the residuals themselves because they're quite large for
    # an overdetermined system. Instead we just check the results between
    # Vector/StaticArray match.
    @test norm(collect(sol_static.u) - sol_vec.u) < 1.0e-6
end
