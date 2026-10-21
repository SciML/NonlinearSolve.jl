using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder

# `SCCNonlinearProblem` has no `u0`/`kwargs` field, so `init` previously fell through to
# `NonlinearSolveBase`'s generic pipeline and raised an opaque `FieldError` (see
# core_tests__item8.jl history / SciML/NonlinearSolve.jl#1382 for the error-only version
# this replaces). `init` now returns an `SCCNonlinearCache` instead, mirroring
# `SteadyStateDiffEq.jl`'s `SteadyStateSCCCache` for the analogous SCC-lowered
# `SteadyStateProblem` case.
function manual_scc_problem()
    function f1(du, u, p)
        du[1] = cos(u[2]) - u[1]
        return du[2] = sin(u[1] + u[2]) + u[2]
    end
    explicitfun1(p, sols) = nothing
    prob1 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), zeros(2), zeros(3)
    )

    function f2(du, u, p)
        du[1] = 2u[2] + u[1] + 1.0
        du[2] = u[3]^2 + u[2]
        return du[3] = u[1]^2 + u[3]
    end
    explicitfun2(p, sols) = nothing
    prob2 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), zeros(3), zeros(3)
    )

    p = zeros(3)
    A3 = [2.0 2.5 1.5; 4.0 -1.5 1.5; 1.0 -1.0 -1.0]
    prob3 = LinearProblem(A3, p, zeros(3))
    function explicitfun3(p, sols)
        p[1] = -(sols[1][1] + sols[1][2] + sols[2][1] + sols[2][2] + sols[2][3])
        p[2] = -(sols[1][1] + sols[1][2] + sols[2][1] + 2.0sols[2][2] + sols[2][3])
        return p[3] = -(
            sols[1][1] + 2.0sols[1][2] + 3.0sols[2][1] + 5.0sols[2][2] +
                6.0sols[2][3]
        )
    end

    return SciMLBase.SCCNonlinearProblem(
        (prob1, prob2, prob3), (explicitfun1, explicitfun2, explicitfun3)
    )
end

@testset "init builds a cache, no eager solve" for alg in (
        SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing),
        NewtonRaphson(),
    )
    sccprob = manual_scc_problem()
    for cache in (init(sccprob, alg), init(sccprob))
        @test cache isa SCCNonlinearSolve.SCCNonlinearCache
        @test isempty(cache.sols)
    end
end

@testset "solve!(init(prob, alg)) matches solve(prob, alg)" begin
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    sccprob = manual_scc_problem()
    sol_direct = solve(sccprob, alg)

    cache = init(sccprob, alg)
    sol_init = solve!(cache)
    @test sol_init.u == sol_direct.u
    @test sol_init.resid == sol_direct.resid
    @test sol_init.retcode == sol_direct.retcode

    # `init(prob)`/`init(prob, NewtonRaphson())` mirror `solve`'s own default-algorithm
    # dispatch chain.
    import NonlinearSolve # Required for Default
    @test solve!(init(sccprob)).u == sol_direct.u
    @test solve!(init(sccprob, NewtonRaphson())).u == sol_direct.u
end

@testset "stepping to completion equals solve!" begin
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    sccprob = manual_scc_problem()
    sol_direct = solve(sccprob, alg)

    cache = init(sccprob, alg)
    for _ in 1:3
        step!(cache)
    end
    @test length(cache.sols) == 3
    sol_stepped = solve!(cache)
    @test sol_stepped.u == sol_direct.u
    @test sol_stepped.resid == sol_direct.resid
    @test sol_stepped.retcode == sol_direct.retcode

    # Partial stepping, then `solve!` finishes the remaining blocks.
    cache2 = init(sccprob, alg)
    step!(cache2)
    @test length(cache2.sols) == 1
    sol_partial = solve!(cache2)
    @test sol_partial.u == sol_direct.u

    # `step!` past completion is a no-op.
    cache3 = init(sccprob, alg)
    for _ in 1:5
        step!(cache3)
    end
    @test length(cache3.sols) == 3
    @test solve!(cache3).u == sol_direct.u
end

@testset "reinit! resets every block" begin
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    sccprob = manual_scc_problem()
    sol_direct = solve(sccprob, alg)

    cache = init(sccprob, alg)
    step!(cache)
    step!(cache)
    @test length(cache.sols) == 2

    SciMLBase.reinit!(cache)
    @test isempty(cache.sols)
    @test solve!(cache).u == sol_direct.u

    # `reinit!` after a full `solve!` (which internally advances the cache to
    # completion) also resets cleanly.
    solve!(cache)
    SciMLBase.reinit!(cache)
    @test isempty(cache.sols)
    @test solve!(cache).u == sol_direct.u
end

@testset "linear and homotopy blocks work through init/step!/solve!" begin
    # Same scenario as core_tests__item5.jl's homotopy test, plus a linear block, so
    # all three block kinds are exercised through the cache.
    Hbranch(u, p, λ) = [(1 - λ) * 2 * (u[1] - 1) + λ * (u[1]^2 - 1)]
    homblock = HomotopyProblem(Hbranch, [-3.0])
    nlblock = NonlinearProblem((u, p) -> [u[1]^2 - 4], [1.5])
    linblock = LinearProblem([2.0;;], [6.0], [0.0])
    sccprob = SciMLBase.SCCNonlinearProblem(
        (linblock, homblock, nlblock),
        SciMLBase.Void{Any}.([Returns(nothing), Returns(nothing), Returns(nothing)])
    )
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    sol_direct = solve(sccprob, alg)
    @test sol_direct.u ≈ [3.0, 1.0, 2.0] atol = 1.0e-6  # linear, homotopy (physical root), plain

    cache = init(sccprob, alg)
    step!(cache)
    step!(cache)
    step!(cache)
    @test solve!(cache).u ≈ sol_direct.u atol = 1.0e-10

    @test solve!(init(sccprob, alg)).u ≈ sol_direct.u atol = 1.0e-10
end
