@testitem "HomotopySweep construction + defaults" tags = [:core] begin
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
end

@testitem "HomotopySweep happy path (oop, λ as separate argument)" tags = [:core] begin
    using SciMLBase
    # f(u,p,λ) = (1-λ)*(u-c) + λ*(u^2-c); λ=0 root u=c ; λ=1 root u=√c.
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    c = 4.0
    prob = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))

    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ sqrt(c) atol = 1.0e-6
end

@testitem "HomotopySweep in-place residual f(du, u, p, λ)" tags = [:core] begin
    using SciMLBase
    H!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)
    prob = HomotopyProblem(H!, [4.0], [4.0])
    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-6
end

@testitem "HomotopySweep with structured (NamedTuple) parameters" tags = [:core] begin
    using SciMLBase
    # λ is not an entry of p anymore, so p needs no particular structure.
    # This construction was impossible under the homotopy_parameter design.
    H(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
    prob = HomotopyProblem(H, [4.0], (c = 4.0,))
    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-6
end

@testitem "HomotopySweep rescues an out-of-basin guess (no MTK)" tags = [:core] begin
    using SciMLBase
    # actual residual atan(u-3) has root u=3 but its derivative saturates, so a cold
    # Newton from u0=12 overshoots/diverges. simplified residual u has root u=0.
    H(u, p, λ) = [(1 - λ) * u[1] + λ * atan(u[1] - 3.0)]
    prob = HomotopyProblem(H, [12.0]; λspan = (0.0, 1.0))

    # nsteps with adaptive=true sets the INITIAL step (span/nsteps); bisection stays on
    sol = solve(prob, HomotopySweep(; nsteps = 20))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1.0e-5

    # contrast: cold Newton on the actual system from the same guess does NOT land on 3.
    cold = NonlinearProblem((u, p) -> [atan(u[1] - 3.0)], [12.0])
    csol = solve(cold, NewtonRaphson())
    @test !(SciMLBase.successful_retcode(csol) && isapprox(csol.u[1], 3.0; atol = 1.0e-3))
end

@testitem "HomotopySweep reports a failure retcode when it cannot finish" tags = [:core] begin
    using SciMLBase
    # No real root for λ>1/3 (fold): (1-λ)*u + λ*(u^2 + 1). Continuation cannot reach λ=1.
    H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
    prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep(; min_dλ = 1.0e-2))     # explicit floor keeps the test fast
    @test !SciMLBase.successful_retcode(sol)              # must fail, not silently "succeed"
    @test sol.retcode != SciMLBase.ReturnCode.Success
    @test all(isfinite, sol.u)    # failure returns the last CONVERGED iterate, not a diverged buffer
end

@testitem "HomotopySweep returns last converged iterate even with aliasing" tags = [:core] begin
    using SciMLBase
    # kwargs (incl. the alias specifier) are forwarded to every inner solve; with
    # aliasing on, the inner solver iterates directly in its u0 buffer, so without
    # copy protection the returned u would be a diverged buffer rather than the
    # last converged iterate. Identical step sequences make exact equality against
    # the no-aliasing reference run valid.
    H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
    prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
    sol = solve(
        prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2);
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)
    )
    ref = solve(prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2))
    @test !SciMLBase.successful_retcode(sol)
    @test sol.u == ref.u    # last CONVERGED iterate, not the aliased diverged buffer
end

@testitem "HomotopySweep honors HomotopyProblem-level solver kwargs" tags = [:core] begin
    using SciMLBase
    # maxiters = 1 stored on the problem must reach the inner solves and wreck them;
    # before prob.kwargs forwarding this succeeded by silently ignoring it.
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    prob = HomotopyProblem(H, [4.0], [4.0]; maxiters = 1)
    sol = solve(prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2))
    @test !SciMLBase.successful_retcode(sol)
end

@testitem "HomotopySweep adaptive=false takes fixed steps and fails fast" tags = [:core] begin
    using SciMLBase
    H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
    prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep(; adaptive = false, nsteps = 10))
    @test !SciMLBase.successful_retcode(sol)
end

@testitem "HomotopySweep stalls (does not hang) when dλ underflows eps(λ)" tags = [:core] begin
    using SciMLBase
    # rootless residual on a large-magnitude span: bisection drives dλ below
    # eps(λ) ≈ 1.2e-7 long before the absolute sqrt(eps) floor stops it, so the
    # stall guard must fire instead of looping forever.
    H(u, p, λ) = [u[1]^2 + 1.0]
    prob = HomotopyProblem(H, [0.0]; λspan = (1.0e9, 2.0e9))
    sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()); maxiters = 5)
    @test sol.retcode == SciMLBase.ReturnCode.Stalled
    @test !SciMLBase.successful_retcode(sol)
    @test sol.resid === nothing
end

@testitem "HomotopySweep handles a decreasing λspan" tags = [:core] begin
    using SciMLBase
    # same family swept 1 → 0: target is the λ=0 root u = c.
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    c = 4.0
    prob = HomotopyProblem(H, [2.0], [c]; λspan = (1.0, 0.0))
    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ c atol = 1.0e-6
end

@testitem "HomotopySweep stays in Float32 (no promotion)" tags = [:core] begin
    using SciMLBase
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    prob = HomotopyProblem(H, [4.0f0], [4.0f0]; λspan = (0.0f0, 1.0f0))
    sol = solve(prob, HomotopySweep())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0f0 atol = 1.0e-3
end

@testitem "HomotopySweep inner solver is composable" tags = [:core] begin
    using SciMLBase
    c = 4.0
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    mkprob() = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))

    # default inner (nothing → polyalgorithm)
    s_default = solve(mkprob(), HomotopySweep())
    @test SciMLBase.successful_retcode(s_default)
    @test s_default.u[1] ≈ sqrt(c) atol = 1.0e-6

    # explicit inner: NewtonRaphson
    s_nr = solve(mkprob(), HomotopySweep(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(s_nr)
    @test s_nr.u[1] ≈ sqrt(c) atol = 1.0e-6

    # explicit inner: SimpleNewtonRaphson (proves no hardcoded dependency on a specific alg)
    s_snr = solve(mkprob(), HomotopySweep(; inner = SimpleNewtonRaphson()))
    @test SciMLBase.successful_retcode(s_snr)
    @test s_snr.u[1] ≈ sqrt(c) atol = 1.0e-6
end

@testitem "HomotopyProblem defaults to HomotopySweep when alg is nothing" tags = [:core] begin
    using SciMLBase
    c = 4.0
    H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
    prob = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))
    sol = solve(prob, nothing)        # no algorithm → should default to HomotopySweep
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ sqrt(c) atol = 1.0e-6

    sol2 = solve(prob)                # zero-arg form must route the same way
    @test SciMLBase.successful_retcode(sol2)
    @test sol2.u[1] ≈ sqrt(c) atol = 1.0e-6
end
