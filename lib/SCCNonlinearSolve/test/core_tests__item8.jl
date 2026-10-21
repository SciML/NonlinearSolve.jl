using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using FunctionWrappers: FunctionWrapper

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

    @test solve!(cache).u == sol_direct.u
    SciMLBase.reinit!(cache)
    @test isempty(cache.sols)
    @test solve!(cache).u == sol_direct.u
end

# A `solve!` on an already-completed cache (whether completed by a prior `solve!`, or
# by `step!`ping/`solve!`ing to the end) must keep returning the *same* solution,
# not recombine an empty/discarded history into a spuriously "successful" empty result.
@testset "repeated solve! after completion returns the same solution" begin
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    sccprob = manual_scc_problem()
    sol_direct = solve(sccprob, alg)

    cache = init(sccprob, alg)
    first_sol = solve!(cache)
    @test length(cache.sols) == 3
    second_sol = solve!(cache)
    @test first_sol.u == second_sol.u == sol_direct.u
    @test first_sol.resid == second_sol.resid == sol_direct.resid
    @test first_sol.retcode == second_sol.retcode == sol_direct.retcode

    # Completion by stepping, then two `solve!` calls (the second after a
    # completion-no-op `step!`).
    cache2 = init(sccprob, alg)
    for _ in 1:3
        step!(cache2)
    end
    stepped_sol = solve!(cache2)
    step!(cache2)
    @test solve!(cache2).u == stepped_sol.u == sol_direct.u
end

# `solve_single_scc`/`solve` pass each block's `explicitfuns!` the solutions of every
# prior block in a specific representation: a `Tuple` for tuple-form blocks, or a typed
# `view` into a concrete `Vector` (so `FunctionWrappers`-typed `explicitfuns!` can
# dispatch) for vector-form blocks. `step!` must pass the identical representation, not
# a generic `Vector{Any}`, or callbacks that rely on either form break.
@testset "stepping preserves the explicitfuns! argument representation" begin
    alg = NewtonRaphson()

    @testset "tuple-specialized callback, tuple-form blocks" begin
        b1 = NonlinearProblem((u, p) -> u .- p, [0.0], [2.0])
        b2 = NonlinearProblem((u, p) -> u .- p, [0.0], [0.0])
        e1(p, sols::Tuple) = nothing
        e2(p, sols::Tuple) = (p[1] = 3sols[1].u[1]; nothing)
        sccprob = SciMLBase.SCCNonlinearProblem((b1, b2), (e1, e2))

        sol_direct = solve(sccprob, alg)
        @test sol_direct.u == [2.0, 6.0] # block 2's p is coupled to 3 * block 1's u

        @test solve!(init(sccprob, alg)).u == sol_direct.u

        cache = init(sccprob, alg)
        step!(cache) # calls e1 with sols::Tuple{} (no prior blocks)
        @test solve!(cache).u == sol_direct.u # the remaining step! calls e2 with sols::Tuple{<:NonlinearSolution}
    end

    @testset "FunctionWrapper-typed callback, vector-form blocks" begin
        b1 = NonlinearProblem((u, p) -> u .- p, [0.0], [2.0])
        b2 = NonlinearProblem((u, p) -> u .- p, [0.0], [0.0])
        probs = [b1, b2]
        # Derive the per-block solution type (and the typed `view` over it) the same
        # way the existing vector-form driver does, rather than hand-spelling
        # `NonlinearSolution`'s type parameters (whose count can change between
        # SciMLBase versions — see SciML/NonlinearSolve.jl#1332).
        ST = SCCNonlinearSolve.__scc_vector_sol_type(probs)
        SV = SubArray{ST, 1, Vector{ST}, Tuple{UnitRange{Int}}, true}
        FW = FunctionWrapper{Nothing, Tuple{Vector{Float64}, SV}}
        explicitfuns = [FW((p, sols) -> nothing), FW((p, sols) -> (p[1] = 3sols[1].u[1]; nothing))]
        sccprob = SciMLBase.SCCNonlinearProblem(probs, explicitfuns)

        sol_direct = solve(sccprob, alg)
        @test sol_direct.u == [2.0, 6.0]

        @test solve!(init(sccprob, alg)).u == sol_direct.u

        cache = init(sccprob, alg)
        step!(cache) # calls the first FunctionWrapper with an empty, but SV-typed, view
        @test solve!(cache).u == sol_direct.u # the remaining step! calls the second with a 1-element SV view
    end
end

# `store_original = Val(true)` records the per-block solutions on the final solution's
# `original` field, in whatever container `solve` itself builds (a `Tuple` for
# tuple-form blocks, a concrete `Vector{<:NonlinearSolution}` for vector-form blocks).
# Stepping must preserve that container type, not normalize everything to a `Tuple`.
@testset "store_original preserves the driver's container and element types" begin
    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), store_original = Val(true))

    @testset "tuple-form blocks" begin
        sccprob = manual_scc_problem()
        direct = solve(sccprob, alg)

        cache = init(sccprob, alg)
        step!(cache)
        stepped = solve!(cache)

        @test typeof(direct) == typeof(stepped)
        @test typeof(direct.original) == typeof(stepped.original)
        @test direct.original == stepped.original
    end

    @testset "vector-form blocks" begin
        b1 = NonlinearProblem((u, p) -> u .- p, [0.0], [2.0])
        b2 = NonlinearProblem((u, p) -> u .- p, [0.0], [0.0])
        e1(p, sols) = nothing
        e2(p, sols) = (p[1] = 3sols[1].u[1]; nothing)
        sccprob = SciMLBase.SCCNonlinearProblem([b1, b2], [e1, e2])
        direct = solve(sccprob, alg)

        cache = init(sccprob, alg)
        step!(cache)
        stepped = solve!(cache)

        @test typeof(direct) == typeof(stepped)
        @test direct.original isa Vector{<:SciMLBase.NonlinearSolution}
        @test typeof(direct.original) == typeof(stepped.original)
        @test direct.original == stepped.original
    end
end

# `reinit!(cache, u0; p)`/`reinit!(cache; p)`/`reinit!(cache, u0)` must match a fresh
# `solve` of `remake(prob; u0, p)`, both right after `init` and after the cache has
# already been (partially or fully) stepped.
@testset "reinit! with replacement u0/p" begin
    alg = NewtonRaphson()
    make_aliased_prob(p = [4.0]) = SciMLBase.SCCNonlinearProblem(
        (
            NonlinearProblem((u, p) -> u .^ 2 .- p[1], [1.0], p),
            NonlinearProblem((u, p) -> u .^ 2 .- 4p[1], [1.0], p),
        ),
        (Returns(nothing), Returns(nothing)), p, true
    )
    new_u0 = [-1.0, -1.0]
    new_p = [9.0]

    @testset "state+parameter reinit!, after complete execution" begin
        sccprob = make_aliased_prob()
        fresh = solve(SciMLBase.remake(sccprob; u0 = new_u0, p = new_p), alg)

        cache = init(sccprob, alg)
        solve!(cache)
        SciMLBase.reinit!(cache, new_u0; p = new_p)
        @test isempty(cache.sols)
        @test solve!(cache).u == fresh.u
    end

    @testset "parameter-only reinit!, after partial execution" begin
        sccprob = make_aliased_prob()
        fresh = solve(SciMLBase.remake(sccprob; p = new_p), alg)

        cache = init(sccprob, alg)
        step!(cache)
        SciMLBase.reinit!(cache; p = new_p)
        @test isempty(cache.sols)
        @test solve!(cache).u == fresh.u
    end

    @testset "state-only reinit!, after partial execution" begin
        sccprob = make_aliased_prob()
        fresh = solve(SciMLBase.remake(sccprob; u0 = new_u0), alg)

        cache = init(sccprob, alg)
        step!(cache)
        SciMLBase.reinit!(cache, new_u0)
        @test isempty(cache.sols)
        @test solve!(cache).u == fresh.u
    end

    @testset "no-arg reinit! after complete execution keeps the problem unchanged" begin
        sccprob = make_aliased_prob()
        fresh = solve(sccprob, alg)

        cache = init(sccprob, alg)
        solve!(cache)
        SciMLBase.reinit!(cache)
        @test isempty(cache.sols)
        @test solve!(cache).u == fresh.u
    end

    @testset "p without parameters_alias errors, matching remake" begin
        sccprob = SciMLBase.SCCNonlinearProblem(
            (
                NonlinearProblem((u, p) -> u .- p, [0.0], [2.0]),
                NonlinearProblem((u, p) -> u .- p, [0.0], [0.0]),
            ),
            (Returns(nothing), Returns(nothing))
        )
        cache = init(sccprob, alg)
        @test_throws ArgumentError SciMLBase.reinit!(cache; p = [4.0])
    end
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
