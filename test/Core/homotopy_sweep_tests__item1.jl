using NonlinearSolve

alg = HomotopySweep()
@test alg isa HomotopySweep
@test alg isa NonlinearSolve.AbstractNonlinearSolveAlgorithm
@test alg.inner === nothing
@test alg.nsteps === nothing
@test alg.adaptive == true
@test alg.initial_step_factor ≈ 0.1
@test alg.min_dλ === nothing            # resolved to sqrt(eps) at solve time

alg2 = HomotopySweep(;
    inner = SimpleNewtonRaphson(), nsteps = 20,
    adaptive = false, min_dλ = 1.0e-4
)
@test alg2.nsteps == 20
@test alg2.adaptive == false
@test alg2.min_dλ ≈ 1.0e-4

# fixed-step mode must be explicit about its resolution
@test_throws ArgumentError HomotopySweep(; adaptive = false)
@test_throws ArgumentError HomotopySweep(; nsteps = 0)
