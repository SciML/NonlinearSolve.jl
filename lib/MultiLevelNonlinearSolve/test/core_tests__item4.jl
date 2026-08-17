using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LinearAlgebra

# T10 — a user `postcondition` corrects the FULL state at the commit point, once per accepted
# iterate. It must not be forwarded into the condensed solve, where it would be handed a
# length-`n̄` vector it was never written for.
@testset "T10 postcondition runs on the full state at the commit point" begin
    @test NonlinearSolveBase.supports_postcondition(MultiLevelNewton())

    seen_lengths = Int[]
    ncalls = Ref(0)
    function corrector!(u, u_prev, p, cache)
        ncalls[] += 1
        push!(seen_lengths, length(u))
        return u        # identity: only the call protocol is under test
    end

    prob, model = bar_problem()
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12, postcondition = corrector!)
    n_full = 4 * model.n

    # The condensed solve never sees it.
    @test !haskey(cache.global_cache.kwargs, :postcondition)

    nsteps = run_steps!(cache; maxsteps = 50)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test all(==(n_full), seen_lengths)
    # Once per accepted iterate, plus the initial-guess correction the conditioning machinery
    # applies to `u0` before any cache exists.
    @test ncalls[] == nsteps + 1
end

@testset "T10 a postcondition that moves ū is respected" begin
    # Clamp the primary block. The corrector runs before `q` is re-committed, so the returned
    # state must be internally consistent: `q` corresponds to the clamped `ū`, not the
    # proposed one.
    cap = 0.30
    clamp_primary!(u, u_prev, p, cache) = (clamp!(view(u, 1:40), -Inf, cap); u)

    prob, model = bar_problem()
    sol = solve(
        prob, MultiLevelNewton(); abstol = 1.0e-12, maxiters = 50,
        postcondition = clamp_primary!
    )
    @test maximum(view(sol.u, 1:40)) ≤ cap + 1.0e-12

    F = zeros(4 * model.n)
    monolithic!(F, sol.u, model)
    # The `q` rows are what the commit is responsible for; they must hold at the corrected ū
    # even though the clamp keeps the ū rows from converging.
    @test maximum(abs, view(F, 41:160)) < 1.0e-10
end

# Stands in for the `SimpleNonlinearSolve` family: no `__init` of its own, so it falls back
# to a cache with no iterator interface.
struct SolveOnlyAlgorithm <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end

# The API surface refuses the configurations whose failure would otherwise be silent.
@testset "rejected configurations" begin
    prob, model = bar_problem()

    # `lb`/`ub` would be handled by wrapping `prob.f.f` in a bounds transform — but on this
    # problem that field is the condensed function the global solver is built from.
    bounded = NonlinearProblem(
        prob.f, prob.u0, prob.p; lb = fill(-10.0, 160), ub = fill(10.0, 160)
    )
    @test_throws ArgumentError init(bounded, MultiLevelNewton())

    # A `precondition` corrector acts on the full residual; the solved system is condensed.
    @test_throws ArgumentError init(prob, MultiLevelNewton(); precondition = identity)

    # A plain `NonlinearFunction` carries no `primary`/`internal` split.
    plain = NonlinearProblem(NonlinearFunction(monolithic!), zeros(160), model)
    @test_throws ArgumentError init(plain, MultiLevelNewton())

    # A solve-only algorithm (every `SimpleNonlinearSolve` one, for instance) defines no
    # `__init` of its own and so builds a cache with no `step!` — but this solver drives the
    # condensed solve one step at a time.
    @test_throws ArgumentError init(
        prob, MultiLevelNewton(; global_solver = SolveOnlyAlgorithm())
    )

    @test_throws ArgumentError MultiLevelNewton(; jacobian_reuse = :sometimes)
    @test_throws ArgumentError LocalToleranceSchedule(; schedule = :cubic)

    # Out-of-place condensed functions are not supported: the elimination is built on
    # mutating workspaces.
    @test_throws ArgumentError MultiLevelNonlinearFunction(
        NonlinearFunction{false}((u, p) -> u); primary = 1:1, internal = 2:2,
        commit_internal! = (q, u, p) -> true
    )
end

# The wrapper deliberately does not advertise the condensed Jacobian at full length: it is
# `n̄ × n̄` while the wrapper's residual has length `n`.
@testset "the wrapper does not forward the condensed Jacobian" begin
    prob, _ = bar_problem()
    @test !SciMLBase.has_jac(prob.f)
    @test !SciMLBase.has_jvp(prob.f)
    @test !SciMLBase.has_vjp(prob.f)
    @test SciMLBase.has_jac(prob.f.f)          # the condensed function does have one
    @test !hasproperty(prob.f, :jac)
    @test !hasproperty(prob.f, :jac_prototype)

    # It is callable at full length, with structurally zero internal rows.
    res = fill(NaN, 160)
    prob.f(res, zeros(160), prob.p)
    @test all(iszero, view(res, 41:160))
    @test !any(isnan, res)
end
