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

@testitem "HomotopySweep rescues an out-of-basin guess (no MTK)" tags = [:core] begin
    using SciMLBase
    # actual residual atan(u-3) has root u=3 but its derivative saturates, so a cold
    # Newton from u0=12 overshoots/diverges. simplified residual u has root u=0.
    # H(u,p) = (1-λ)*u + λ*atan(u-3); p=[λ]. Sweep 0→1 tracks 0 → 3.
    H(u, p) = [(1 - p[1]) * u[1] + p[1] * atan(u[1] - 3.0)]
    u0 = [12.0]
    p = [0.0]
    prob = HomotopyProblem(H, u0, p; homotopy_parameter = 1, λspan = (0.0, 1.0))

    sol = solve(prob, HomotopySweep(; nsteps = 20))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1e-5

    # contrast: cold Newton on the actual system from the same guess does NOT land on 3.
    cold = NonlinearProblem((u, p) -> [atan(u[1] - 3.0)], u0)
    csol = solve(cold, NewtonRaphson())
    @test !(SciMLBase.successful_retcode(csol) && isapprox(csol.u[1], 3.0; atol = 1e-3))
end

@testitem "HomotopySweep reports a failure retcode when it cannot finish" tags = [:core] begin
    using SciMLBase
    # No real root for λ>1/3 (fold): (1-λ)*u + λ*(u^2 + 1). Continuation cannot reach λ=1.
    H(u, p) = [(1 - p[1]) * u[1] + p[1] * (u[1]^2 + 1.0)]
    prob = HomotopyProblem(H, [0.0], [0.0]; homotopy_parameter = 1, λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep(; adaptive = true, min_dλ = 1e-2))
    @test !SciMLBase.successful_retcode(sol)            # must fail, not silently "succeed"
    @test sol.retcode != SciMLBase.ReturnCode.Success
end

@testitem "HomotopySweep adaptive=false fails fast on a hard step" tags = [:core] begin
    using SciMLBase
    H(u, p) = [(1 - p[1]) * u[1] + p[1] * (u[1]^2 + 1.0)]
    prob = HomotopyProblem(H, [0.0], [0.0]; homotopy_parameter = 1, λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep(; adaptive = false))
    @test !SciMLBase.successful_retcode(sol)
end

@testitem "HomotopySweep inner solver is composable" tags = [:core] begin
    using SciMLBase
    c = 4.0
    H(u, p) = [(1 - p[2]) * (u[1] - c) + p[2] * (u[1]^2 - c)]
    mkprob() = HomotopyProblem(H, [c], [c, 0.0]; homotopy_parameter = 2, λspan = (0.0, 1.0))

    # default inner (nothing → polyalgorithm)
    s_default = solve(mkprob(), HomotopySweep())
    @test SciMLBase.successful_retcode(s_default)
    @test s_default.u[1] ≈ sqrt(c) atol = 1e-6

    # explicit inner: NewtonRaphson
    s_nr = solve(mkprob(), HomotopySweep(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(s_nr)
    @test s_nr.u[1] ≈ sqrt(c) atol = 1e-6

    # explicit inner: SimpleNewtonRaphson (proves no hardcoded dependency on a specific alg)
    s_snr = solve(mkprob(), HomotopySweep(; inner = SimpleNewtonRaphson()))
    @test SciMLBase.successful_retcode(s_snr)
    @test s_snr.u[1] ≈ sqrt(c) atol = 1e-6
end

@testitem "HomotopySweep handles integer-eltype p" tags = [:core] begin
    using SciMLBase
    # p is Vector{Int}; the λ-setter must promote so a Float λ doesn't throw InexactError.
    c = 4.0
    H(u, p) = [(1 - p[2]) * (u[1] - c) + p[2] * (u[1]^2 - c)]
    u0 = [c]
    p = [4, 0]                      # Vector{Int}; index 2 is λ
    prob = HomotopyProblem(H, u0, p; homotopy_parameter = 2, λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ sqrt(c) atol = 1e-6
end

@testitem "HomotopyProblem defaults to HomotopySweep when alg is nothing" tags = [:core] begin
    using SciMLBase
    c = 4.0
    H(u, p) = [(1 - p[2]) * (u[1] - c) + p[2] * (u[1]^2 - c)]
    prob = HomotopyProblem(H, [c], [c, 0.0]; homotopy_parameter = 2, λspan = (0.0, 1.0))
    sol = solve(prob, nothing)        # no algorithm → should default to HomotopySweep
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ sqrt(c) atol = 1e-6
end
