using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using ADTypes: AutoForwardDiff
using LinearAlgebra

# T1 — the eliminated solve finds the same full-length root as a monolithic solve of the
# unreduced problem. Run on the heterogeneous bar too, where every element has a different
# tangent, so a bug that silently assumes a uniform `S` cannot pass.
@testset "T1 root matches the monolithic twin — $(label)" for (label, cscale) in
                                                             (
        ("homogeneous", ones(40)), ("heterogeneous", HETEROGENEOUS_CSCALE),
    )
    prob, model = bar_problem(; cscale)
    sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(sol)
    @test length(sol.u) == 4 * model.n

    mono = solve(
        monolithic_problem(model), NewtonRaphson(; autodiff = AutoForwardDiff());
        abstol = 1.0e-12
    )
    @test SciMLBase.successful_retcode(mono)
    @test maximum(abs, sol.u .- mono.u) < 1.0e-10

    # The `q` rows of the residual are structurally zero: `q` was eliminated, so its
    # equations hold by construction and `sol.resid` stays comparable with the monolithic one.
    @test all(iszero, view(sol.resid, (model.n + 1):(4 * model.n)))
    @test maximum(abs, view(sol.resid, 1:model.n)) < 1.0e-10
end

# T2 — the residual of the *unreduced* problem at the multi-level root. Without a local
# forcing schedule the local solves run at a fixed tight tolerance and the bound is tight;
# with one, the committed `q` is only accurate to the schedule's floor, so the reachable
# threshold is `abstol + C·floor` rather than `abstol`.
@testset "T2 monolithic residual at the multi-level root" begin
    prob, model = bar_problem()
    sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-12)
    F = zeros(4 * model.n)
    monolithic!(F, sol.u, model)
    @test maximum(abs, F) < 1.0e-11

    # With a schedule the committed `q` is only as accurate as the schedule's floor, so the
    # reachable bound is `abstol + C·floor`, not `abstol`.
    schedule = LocalToleranceSchedule(; floor_rel = 1.0e-2)
    prob_s, model_s = bar_problem(; local_tolerance = schedule)
    abstol = 1.0e-10
    sol_s = solve(prob_s, MultiLevelNewton(); abstol)
    @test SciMLBase.successful_retcode(sol_s)
    Fs = zeros(4 * model_s.n)
    monolithic!(Fs, sol_s.u, model_s)
    @test maximum(abs, Fs) < abstol + 10 * schedule.floor_rel * abstol

    # That the floor is what limits it, and not `abstol`: a schedule pinned at a deliberately
    # sloppy tolerance leaves a monolithic residual of the order of that tolerance, even
    # though the condensed solve reports success at the same `abstol`.
    sloppy = LocalToleranceSchedule(; schedule = :fixed, tol_init = 1.0e-4)
    prob_l, model_l = bar_problem(; local_tolerance = sloppy)
    sol_l = solve(prob_l, MultiLevelNewton(); abstol)
    Fl = zeros(4 * model_l.n)
    monolithic!(Fl, sol_l.u, model_l)
    @test maximum(abs, Fl) > 1.0e3 * maximum(abs, Fs)
end
