using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LinearAlgebra

# T15 — a `:fixed` schedule pinned at the tolerance the fixture uses by default must produce
# exactly the run you get with no schedule at all. This is what makes the local-forcing
# machinery opt-in rather than a behaviour change.
@testset "T15 a fixed schedule reproduces the unscheduled run" begin
    abstol = 1.0e-12
    prob_none, model_none = bar_problem()
    e_none = residual_history(init(prob_none, MultiLevelNewton(); abstol))

    prob_fix, model_fix = bar_problem(;
        local_tolerance = LocalToleranceSchedule(;
            schedule = :fixed, tol_init = 1.0e-14, floor_rel = 1.0e-2
        )
    )
    e_fix = residual_history(init(prob_fix, MultiLevelNewton(); abstol))

    @test e_none == e_fix
end

# T11 — an adaptive schedule does strictly less local work than solving every local problem
# to full precision from the first iteration, and stays superlinear while doing it.
#
# It is *superlinearity* that is asserted, not an order above 1.8. On this fixture the
# scheduled run costs one extra global iteration and its fitted tail order lands near 1.3:
# the bar converges from ‖R̄‖ = 0.3 in four iterations, so there is no asymptotic window in
# which an order could be measured, and the local Newton's own quadratic convergence means a
# requested tolerance is only loosely related to the accuracy actually delivered. The
# quadratic-rate assertion lives in T3, on the unscheduled configuration.
@testset "T11 adaptive local forcing cuts local work and stays superlinear" begin
    abstol = 1.0e-12
    prob_fix, model_fix = bar_problem(;
        local_tolerance = LocalToleranceSchedule(;
            schedule = :fixed, tol_init = 1.0e-12, floor_rel = 1.0e-2
        )
    )
    cache_fix = init(prob_fix, MultiLevelNewton(); abstol)
    e_fix = residual_history(cache_fix)

    prob_ad, model_ad = bar_problem(;
        local_tolerance = LocalToleranceSchedule(; schedule = :quadratic)
    )
    cache_ad = init(prob_ad, MultiLevelNewton(); abstol)
    e_ad = residual_history(cache_ad)

    @test SciMLBase.successful_retcode(cache_ad.retcode)
    @test is_superlinear(e_ad)
    @test length(e_ad) ≤ length(e_fix) + 1
    @test model_ad.counters.nlocaliter < model_fix.counters.nlocaliter
    # Same root, to the accuracy the schedule's floor allows.
    @test maximum(abs, NonlinearSolveBase.get_u(cache_ad) .-
                       NonlinearSolveBase.get_u(cache_fix)) < 1.0e-9

    # A frozen Jacobian is the contrast: linear, so its ratios stop shrinking.
    prob_ch, _ = bar_problem()
    e_ch = residual_history(
        init(prob_ch, MultiLevelNewton(; jacobian_reuse = :chord); abstol); chord_after = 1
    )
    @test !is_superlinear(e_ch)
end

# T12 — a local tolerance far looser than the global one makes the condensed residual itself
# inaccurate, so the solver must not report success at a tolerance it cannot actually have
# reached.
@testset "T12 a sloppy local tolerance does not produce a false success" begin
    prob, model = bar_problem(;
        local_tolerance = LocalToleranceSchedule(;
            schedule = :fixed, tol_init = 1.0e-2, floor_rel = 1.0
        )
    )
    sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-12, maxiters = 30)
    @test !SciMLBase.successful_retcode(sol)
    # Specifically the demotion branch: the condensed solve *did* report success, and
    # re-measuring through the tightened elimination is what takes it back.
    @test sol.retcode == ReturnCode.Stalled
    # And the reported residual is the honest one — measured through the tightened
    # elimination, not the loose one that made the solve look converged.
    @test maximum(abs, view(sol.resid, 1:model.n)) > 1.0e-12
end

# The tolerance cell is owned by the cache, not by the problem: two caches built from the
# same problem must not read or write each other's local tolerance. Without this, two solves
# of one problem object would silently change each other's local accuracy, and a trial
# residual would stop being reproducible.
@testset "the local tolerance cell is per cache" begin
    prob, _ = bar_problem(; local_tolerance = LocalToleranceSchedule())
    a = init(prob, MultiLevelNewton(); abstol = 1.0e-12)
    b = init(prob, MultiLevelNewton(); abstol = 1.0e-12)

    @test a.local_tol !== b.local_tol
    @test a.local_tol[] == b.local_tol[]

    before = b.local_tol[]
    for _ in 1:3
        step!(a)
    end
    @test a.local_tol[] != before      # the schedule really did move
    @test b.local_tol[] == before      # and it did not move the other cache's

    # `local_tolerance(p)` is how user code reads it, and it must see the cache's own cell.
    @test local_tolerance(a.global_cache.p) == a.local_tol[]
    @test local_tolerance(b.global_cache.p) == b.local_tol[]
    # With no schedule the parameters are passed through untouched.
    prob_none, model_none = bar_problem()
    cache_none = init(prob_none, MultiLevelNewton())
    @test cache_none.global_cache.p === model_none
    @test local_tolerance(model_none) === nothing
    @test user_parameters(model_none) === model_none
end
