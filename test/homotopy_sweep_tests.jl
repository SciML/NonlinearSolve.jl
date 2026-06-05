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

@testitem "HomotopySweep happy path (oop, λ in p by index)" tags = [:core] begin
    using SciMLBase
    # H(u,p) = (1-λ)*(u-c) + λ*(u^2-c); p = [c, λ]; index 2 is λ.
    # λ=0 root u=c ; λ=1 root u=√c. Continuation tracks c → √c.
    c = 4.0
    H(u, p) = [(1 - p[2]) * (u[1] - c) + p[2] * (u[1]^2 - c)]
    u0 = [c]                       # start at simplified root
    p = [c, 0.0]
    prob = HomotopyProblem(H, u0, p; homotopy_parameter = 2, λspan = (0.0, 1.0))

    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ sqrt(c) atol = 1e-6
end
