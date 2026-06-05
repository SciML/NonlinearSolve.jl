@testitem "HomotopySweep construction + defaults" tags = [:core] begin
    alg = HomotopySweep()
    @test alg isa HomotopySweep
    @test alg isa NonlinearSolve.AbstractNonlinearSolveAlgorithm
    @test alg.inner === nothing
    @test alg.nsteps == 10
    @test alg.adaptive == true
    @test alg.min_dλ ≈ 1e-3

    alg2 = HomotopySweep(; inner = SimpleNewtonRaphson(), nsteps = 20, adaptive = false, min_dλ = 1e-4)
    @test alg2.nsteps == 20
    @test alg2.adaptive == false
    @test alg2.min_dλ ≈ 1e-4
end
