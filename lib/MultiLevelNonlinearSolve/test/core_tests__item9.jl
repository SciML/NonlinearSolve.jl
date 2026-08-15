using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LineSearch: BackTracking
using LinearSolve: KrylovJL_GMRES, LUFactorization
using LinearAlgebra, SparseArrays
const SciMLOperators = MultiLevelNonlinearSolve.SciMLOperators

# T9 — the full-space arm reaches the same root as the condensed one, and the internal block
# of the step is zero at every iteration. Both properties are conditional on the pieces being
# used together: a plain Newton (no globalization) with `CondensedFactorization` doing the
# δq-zeroing and `MultiLevelProjection` running the commit.
@testset "T9 the full-space arm matches MultiLevelNewton — $(label)" for (label, cscale) in
                                                                        (
        ("homogeneous", ones(40)), ("heterogeneous", HETEROGENEOUS_CSCALE),
    )
    reference, _ = bar_problem(; cscale)
    ref = solve(reference, MultiLevelNewton(); abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(ref)

    mlnf, model = let (prob, model) = bar_problem(; cscale)
        prob.f, model
    end
    prob = fullspace_problem(mlnf, zeros(4 * model.n), model)
    @test prob.f isa NonlinearFunction        # a plain function, not the wrapper type
    @test prob.f.jac_prototype isa SchurOperator
    @test size(prob.f.jac_prototype) == (4 * model.n, 4 * model.n)

    cache = init(
        prob, NewtonRaphson(; linsolve = CondensedFactorization()); abstol = 1.0e-12,
        postcondition = MultiLevelProjection(mlnf)
    )
    internal_steps = Float64[]
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < 50
        k += 1
        step!(cache)
        push!(internal_steps, maximum(abs, view(SciMLBase.get_du(cache), 41:160)))
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    @test all(iszero, internal_steps)         # δu[q] == 0 at every step
    @test maximum(abs, NonlinearSolveBase.get_u(cache) .- ref.u) < 1.0e-9
end

@testset "the projection commits the internal block" begin
    prob_b, model = bar_problem()
    mlnf = prob_b.f
    projection = MultiLevelProjection(mlnf)

    u = zeros(4 * model.n)
    u[1:40] .= range(0.01, 0.42; length = 40)
    projection(u, copy(u), model, nothing)

    # `q` now solves the local problems at this `ū`, which is exactly what makes the internal
    # rows of the full residual vanish.
    F = zeros(4 * model.n)
    monolithic!(F, u, model)
    @test maximum(abs, view(F, 41:160)) < 1.0e-12

    # Idempotent: committing again at the same `ū` changes nothing.
    before = copy(u)
    projection(u, copy(u), model, nothing)
    @test u == before
end

# Globalization is out of scope on this arm and must say so rather than quietly returning a
# step whose internal block is not zero.
@testset "globalization is refused on the full-space arm" begin
    prob_b, model = bar_problem()
    mlnf = prob_b.f
    prob = fullspace_problem(mlnf, zeros(4 * model.n), model)

    cache = init(
        prob,
        NewtonRaphson(; linsolve = CondensedFactorization(), linesearch = BackTracking());
        abstol = 1.0e-12, postcondition = MultiLevelProjection(mlnf)
    )
    @test_throws ArgumentError step!(cache)

    cache_tr = init(
        prob, TrustRegion(; linsolve = CondensedFactorization());
        abstol = 1.0e-12, postcondition = MultiLevelProjection(mlnf)
    )
    @test_throws ArgumentError step!(cache_tr)
end

@testset "fullspace_problem rejects a condensed function it cannot use" begin
    _, model = bar_problem()
    n = model.n
    no_jac = MultiLevelNonlinearFunction(
        NonlinearFunction(Rbar!); primary = 1:n, internal = (n + 1):(4n), commit_internal!
    )
    @test_throws ArgumentError fullspace_problem(no_jac, zeros(4n), model)

    no_proto = MultiLevelNonlinearFunction(
        NonlinearFunction(Rbar!; jac = assemble_S!);
        primary = 1:n, internal = (n + 1):(4n), commit_internal!
    )
    @test_throws ArgumentError fullspace_problem(no_proto, zeros(4n), model)
end

# T5, full-space half — `precs` on the *inner* algorithm has to reach LinearSolve. It is read
# only from the algorithm handed to `init`, so this is only true because
# `CondensedFactorization` builds a real nested cache with `alg.inner` as its top-level
# algorithm. A wrapper that only forwarded `solve!` would degrade preconditioning to the
# identity in silence.
@testset "T5 precs resolves through CondensedFactorization" begin
    nbuilds = Ref(0)
    seen_size = Ref((0, 0))
    function spy_precs(A, p)
        nbuilds[] += 1
        seen_size[] = size(A)
        return (Diagonal(diag(A)), I)
    end

    prob_b, model = bar_problem()
    mlnf = prob_b.f
    prob = fullspace_problem(mlnf, zeros(4 * model.n), model)
    cache = init(
        prob,
        NewtonRaphson(;
            linsolve = CondensedFactorization(; inner = KrylovJL_GMRES(; precs = spy_precs))
        );
        abstol = 1.0e-12, postcondition = MultiLevelProjection(mlnf)
    )
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < 50
        k += 1
        step!(cache)
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    @test nbuilds[] > 0
    # The preconditioner is built from the *condensed* matrix, not the full-size operator.
    @test seen_size[] == (model.n, model.n)
    # One build at the inner `init`, then one per Schur assembly.
    @test nbuilds[] == model.counters.nassembly + 1
end

# T17, second half — Eisenstat–Walker calls `update_tolerances!` unconditionally, and the
# default hook throws for an algorithm that does not define it. The delegation has to carry
# the tolerances into the inner Krylov cache, where they are actually used.
@testset "T17 Eisenstat-Walker forcing works through CondensedFactorization" begin
    prob_b, model = bar_problem()
    mlnf = prob_b.f
    prob = fullspace_problem(mlnf, zeros(4 * model.n), model)
    alg = NewtonRaphson(;
        linsolve = CondensedFactorization(; inner = KrylovJL_GMRES()),
        forcing = EisenstatWalkerForcing2()
    )
    cache = init(
        prob, alg; abstol = 1.0e-10, maxiters = 50,
        postcondition = MultiLevelProjection(mlnf)
    )

    inner_reltols = Float64[]
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < 50
        k += 1
        step!(cache)                      # must not throw
        push!(
            inner_reltols,
            cache.descent_cache.lincache.lincache.cacheval.inner_cache.reltol
        )
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    # The forcing term reached the inner cache and moved, rather than being swallowed.
    @test length(unique(inner_reltols)) > 1
end

# The operator stands for a matrix that is never formed, and it says so. A configuration that
# demands a concrete Jacobian is refused at `init` rather than being handed an operator whose
# `convert` would fail later, or — worse — silently accepted because the convertibility trait
# defaulted to `true` for an operator carrying no sub-operators.
@testset "asking for a concrete Jacobian is refused" begin
    prob_b, model = bar_problem()
    prob = fullspace_problem(prob_b.f, zeros(4 * model.n), model)
    @test !SciMLOperators.isconvertible(prob.f.jac_prototype)
    @test_throws ArgumentError convert(AbstractMatrix, prob.f.jac_prototype)
    @test_throws ArgumentError init(
        prob, NewtonRaphson(; linsolve = CondensedFactorization(), concrete_jac = true);
        postcondition = MultiLevelProjection(prob_b.f)
    )
end

# Subtyping `AbstractFactorization` is what makes the Jacobian-reuse signal reach the linear
# solver: on a reused Jacobian the cache is never marked stale, so neither the inner
# factorization nor its preconditioner is rebuilt. A plain `SciMLLinearSolveAlgorithm` would
# take the generic update path instead, refactorize every step, and never move `nfactors` —
# leaving a reuse test to pass while the saving it measures does not happen.
@testset "jacobian reuse reaches the inner factorization" begin
    prob_b, model = bar_problem()
    mlnf = prob_b.f
    prob = fullspace_problem(mlnf, zeros(4 * model.n), model)
    cache = init(
        prob, NewtonRaphson(; linsolve = CondensedFactorization()); abstol = 1.0e-12,
        maxiters = 100, postcondition = MultiLevelProjection(mlnf)
    )

    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < 100
        k += 1
        step!(cache; recompute_jacobian = k <= 1)     # assemble once, then freeze
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    @test model.counters.nassembly == 1
    @test cache.stats.njacs == 1
    @test cache.stats.nfactors == 1                  # one factorization, not one per step
    @test k > 5                                      # and it really did take many steps
end
