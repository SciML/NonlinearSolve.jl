using LinearAlgebra, NonlinearSolve, PETSc, Test

@testset "PETScSNES return code follows SNES converged reason" begin
    f(u, p) = [u[1]^2 - 2.0]
    petsclib = PETSc.petsclibs[findfirst(x -> x isa PETSc.PetscLibType{Float64}, PETSc.petsclibs)]

    # Issue #1318 MWE: SNES stops with CONVERGED_FNORM_RELATIVE at a residual
    # above abstol (snes_rtol is relative to the initial residual); the wrapper
    # must still report Success from that positive converged reason.
    u0 = [1000.0]
    abstol = 1.0e-8
    reltol = 1.0e-8
    sol = solve(NonlinearProblem(f, u0), PETScSNES(); abstol, reltol)
    @test sol.retcode == ReturnCode.Success
    @test PETSc.LibPETSc.SNESGetConvergedReason(petsclib, sol.original) ==
        PETSc.LibPETSc.SNES_CONVERGED_FNORM_RELATIVE
    @test abstol < norm(sol.resid) <= reltol * norm(f(u0, nothing))
    @test isapprox(sol.u[1], sqrt(2); rtol = 1.0e-3)

    # A solve that reaches abstol still reports success.
    sol = solve(NonlinearProblem(f, [2.0]), PETScSNES(); abstol = 1.0e-8, reltol = 1.0e-8)
    @test sol.retcode == ReturnCode.Success
    @test maximum(abs, sol.resid) ≤ 1.0e-8

    # Exhausting the iteration budget maps SNES_DIVERGED_MAX_IT to MaxIters.
    sol = solve(
        NonlinearProblem(f, [1000.0]), PETScSNES();
        abstol = 1.0e-8, reltol = 1.0e-8, maxiters = 1
    )
    @test sol.retcode == ReturnCode.MaxIters
    @test PETSc.LibPETSc.SNESGetConvergedReason(petsclib, sol.original) ==
        PETSc.LibPETSc.SNES_DIVERGED_MAX_IT

    # Exhausting the function-evaluation budget maps
    # SNES_DIVERGED_FUNCTION_COUNT to MaxIters.
    sol = solve(
        NonlinearProblem(f, [1000.0]), PETScSNES(; snes_max_funcs = 1);
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test sol.retcode == ReturnCode.MaxIters
    @test PETSc.LibPETSc.SNESGetConvergedReason(petsclib, sol.original) ==
        PETSc.LibPETSc.SNES_DIVERGED_FUNCTION_COUNT

    # A full Newton step from u0 = 0.5 grows the residual 1.75x, tripping the
    # divergence tolerance: SNES_DIVERGED_DTOL maps to ConvergenceFailure.
    alg = PETScSNES(; snes_linesearch_type = "basic", snes_divergence_tolerance = 1.5)
    sol = solve(NonlinearProblem(f, [0.5]), alg; abstol = 1.0e-8, reltol = 1.0e-8)
    @test sol.retcode == ReturnCode.ConvergenceFailure
end
