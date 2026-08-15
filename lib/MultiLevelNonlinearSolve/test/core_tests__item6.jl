using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LinearAlgebra

# T13 — the trial/commit protocol. A trial evaluation reads committed internal state and
# writes only scratch, so the condensed residual at a given `ū` does not depend on which
# trials ran before it. Without that, a line search's `ϕ(α)` would drift as it backtracks and
# the step it accepts would depend on the path taken to it.
@testset "T13 a trial residual is reproducible" begin
    prob, model = bar_problem()
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12)
    gc = cache.global_cache
    ū = copy(NonlinearSolveBase.get_u(gc))

    first, elsewhere, again = zeros(40), zeros(40), zeros(40)
    gc.prob.f(first, ū .+ 0.05, gc.p)
    gc.prob.f(elsewhere, ū .+ 0.7, gc.p)      # a far-away trial in between
    gc.prob.f(again, ū .+ 0.05, gc.p)
    @test first == again
    @test first != elsewhere

    # And the committed state is untouched by all of it.
    committed = copy(model.buffer.committed)
    gc.prob.f(elsewhere, ū .+ 0.3, gc.p)
    @test model.buffer.committed == committed
end

@testset "T13 the committed state equals a cold re-solve at the root" begin
    prob, model = bar_problem()
    sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-12)
    ū = sol.u[1:model.n]

    _, cold = bar_problem()                    # a model whose buffer starts at zero
    q_cold = zeros(3 * cold.n)
    @test commit_internal!(q_cold, ū, cold)
    @test maximum(abs, q_cold .- view(sol.u, (model.n + 1):(4 * model.n))) < 1.0e-12
end

# T7 — the ensemble helper. Chunks, not threads, are the unit of work, so nothing is indexed
# by `threadid()` (a task can migrate mid-run) and the partition is fixed at construction.
@testset "T7 the ensemble partitions the points, not the threads" begin
    for (npoints, nchunks) in ((40, 1), (40, 7), (40, 64), (100_000, 8), (3, 8))
        ens = LocalEnsemble(npoints; nchunks)
        @test vcat(ens.chunks...) == collect(1:npoints)
        @test length(ens.chunks) ≤ max(npoints, 1)
        @test all(!isempty, ens.chunks)
    end
end

@testset "T7 threaded local solves are deterministic" begin
    # Big enough that the chunking is real work rather than a formality. Assembly of the
    # condensed residual stays serial — regrouping a scatter-add over shared degrees of
    # freedom would change the rounding, so bitwise agreement with a serial run is not
    # something a threaded reduction can promise.
    n_el = 20_000
    runs = map(1:3) do _
        prob, model = bar_problem(; n_el, threaded = true, nchunks = 8)
        sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        sol.u
    end
    @test runs[1] == runs[2] == runs[3]

    prob_s, _ = bar_problem(; n_el, threaded = false)
    serial = solve(prob_s, MultiLevelNewton(); abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(serial)
    @test runs[1] ≈ serial.u

    # Structural: the per-point workspaces belong to the user's parameters, not to the
    # solver cache. The cache holds the full state and the condensed subcache, nothing sized
    # by the number of quadrature points.
    prob, _ = bar_problem(; n_el = 40)
    cache = init(prob, MultiLevelNewton())
    @test !any(
        n -> getfield(cache, n) isa LocalStateBuffer, fieldnames(typeof(cache))
    )
    @test !any(n -> getfield(cache, n) isa LocalEnsemble, fieldnames(typeof(cache)))
end

@testset "the state buffer keeps trials off the committed state" begin
    buffer = LocalStateBuffer(zeros(3, 4))
    buffer.committed[:, 2] .= [1.0, 2.0, 3.0]

    trial = trial_state(buffer, 2)
    @test trial == [1.0, 2.0, 3.0]        # warm-started from committed
    trial .= [9.0, 9.0, 9.0]
    @test committed_state(buffer, 2) == [1.0, 2.0, 3.0]

    commit_local_state!(buffer)
    @test committed_state(buffer, 2) == [9.0, 9.0, 9.0]
end
