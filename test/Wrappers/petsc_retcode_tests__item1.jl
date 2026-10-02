using LinearAlgebra, NonlinearSolve, PETSc, Test

@testset "PETScSNES return code follows SNES converged reason" begin
    f(u, p) = [u[1]^2 - 2.0]

    # SNES stops with CONVERGED_FNORM_RELATIVE at a residual above abstol; the
    # wrapper must still report success.
    sol = solve(NonlinearProblem(f, [1000.0]), PETScSNES(); abstol = 1e-8, reltol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    @test isapprox(sol.u[1], sqrt(2); rtol = 1e-3)

    # A solve that reaches abstol still reports success.
    sol = solve(NonlinearProblem(f, [2.0]), PETScSNES(); abstol = 1e-8, reltol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    @test maximum(abs, sol.resid) ≤ 1e-8

    # Exhausting the iteration budget maps SNES_DIVERGED_MAX_IT to MaxIters.
    sol = solve(
        NonlinearProblem(f, [1000.0]), PETScSNES(); abstol = 1e-8, reltol = 1e-8, maxiters = 1
    )
    @test sol.retcode == ReturnCode.MaxIters

    # A full Newton step from u0 = 0.5 grows the residual 1.75x, tripping the
    # divergence tolerance: SNES_DIVERGED_DTOL maps to ConvergenceFailure.
    alg = PETScSNES(; snes_linesearch_type = "basic", snes_divergence_tolerance = 1.5)
    sol = solve(NonlinearProblem(f, [0.5]), alg; abstol = 1e-8, reltol = 1e-8)
    @test sol.retcode == ReturnCode.ConvergenceFailure
end
