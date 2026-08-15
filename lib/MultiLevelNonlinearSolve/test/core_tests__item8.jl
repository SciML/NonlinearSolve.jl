using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LineSearch: BackTracking
using LinearAlgebra

# T6 — what a diverged local solve does to the global one. The fixture makes the local
# problem unsolvable beyond a strain threshold (the analogue of a return map leaving its
# admissible region) and the first global step overshoots past it.
#
# The residual rows of a failed trial must be `Inf` and never `NaN`: a backtracking line
# search halves the step on a non-finite merit and recovers, but `NaN` propagates through the
# comparison that decides that, and the search burns its whole budget evaluating at `NaN`.
@testset "T6 a diverged local solve is recoverable by a line search" begin
    prob, model = bar_problem(; corrector_scale = 1.5, fail_above = 0.44)
    sol = solve(
        prob, MultiLevelNewton(;
            global_solver = NewtonRaphson(; linesearch = BackTracking())
        ); abstol = 1.0e-10, maxiters = 40
    )
    @test SciMLBase.successful_retcode(sol)
    @test !any(isnan, sol.u)
    @test !any(isnan, sol.resid)

    reference, _ = bar_problem()
    @test isapprox(
        sol.u[40], solve(reference, MultiLevelNewton(); abstol = 1.0e-12).u[40];
        atol = 1.0e-8
    )
end

@testset "T6 without globalization the failure is reported, not hidden" begin
    prob, model = bar_problem(; corrector_scale = 1.5, fail_above = 0.44)
    sol = solve(prob, MultiLevelNewton(); abstol = 1.0e-10, maxiters = 40)
    @test !SciMLBase.successful_retcode(sol)
    @test sol.retcode == ReturnCode.Unstable      # the `Inf` reaches the convergence test
    @test !any(isnan, sol.u)
    @test !any(isnan, sol.resid)
end

@testset "T6 a commit that reports failure ends the solve" begin
    n = 40
    failing_commit!(q, ū, p) = (commit_internal!(q, ū, p); false)
    model = BarModel()
    f = MultiLevelNonlinearFunction(
        NonlinearFunction(Rbar!; jac = assemble_S!, jac_prototype = sparse_prototype(n));
        primary = 1:n, internal = (n + 1):(4n), commit_internal! = failing_commit!
    )
    sol = solve(NonlinearProblem(f, zeros(4n), model), MultiLevelNewton(); abstol = 1.0e-12)
    @test sol.retcode == ReturnCode.ConvergenceFailure
    @test !any(isnan, sol.u)
end

# `reinit!` has to slice the full-length state before it reaches the condensed cache, which
# iterates on `ū` alone, and re-commit the internal variables at the new starting point.
@testset "reinit! restarts from a new full-length state" begin
    prob, model = bar_problem()
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12)
    solve!(cache)
    root = copy(NonlinearSolveBase.get_u(cache))

    u0 = zeros(4 * model.n)
    u0[1:(model.n)] .= range(0.1, 0.2; length = model.n)
    SciMLBase.reinit!(cache, u0)

    @test cache.nsteps == 0
    @test cache.retcode == ReturnCode.Default
    @test NonlinearSolveBase.get_u(cache)[1:(model.n)] == u0[1:(model.n)]
    @test length(NonlinearSolveBase.get_u(cache.global_cache)) == model.n

    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.u .- root) < 1.0e-9
end

@testset "counters report condensed work, not local work" begin
    prob, model = bar_problem()
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12)
    @test ncommits(cache) == 1                     # the commit at `init`
    step!(cache)
    @test ncommits(cache) == 2
    @test cache.stats.njacs == model.counters.nassembly
    # One `NLStats` is shared with the condensed cache rather than copied between them.
    @test cache.stats === cache.global_cache.stats
    @test cache.stats.nsteps == cache.nsteps
end

@testset "the trace records the full-length iterate" begin
    prob, model = bar_problem()
    sol = solve(
        prob, MultiLevelNewton(); abstol = 1.0e-12, store_trace = Val(true),
        trace_level = TraceAll()
    )
    entries = sol.trace.history
    @test length(entries) ≥ sol.stats.nsteps
    @test length(last(entries).u) == 4 * model.n
    # Iterations are numbered, not all labelled 1 — the symptom of a subcache whose step
    # counter never advances.
    @test length(unique(e.iteration for e in entries)) == length(entries)
end

# The solver's own per-step overhead, measured on a problem whose callbacks allocate nothing
# so that what is left is the cache bookkeeping: the state slices, the commit call and the
# residual/Jacobian mirroring. Views into `u`/`fu` are the thing at risk here — one escaping
# `SubArray` per step would show up immediately.
@testset "step! does not allocate beyond the user's callbacks" begin
    struct Tiny
        q::Vector{Float64}
        scratch::Vector{Float64}
    end
    function tiny_R!(r, u, p)
        for i in eachindex(r)
            p.scratch[i] = 0.5 * tanh(u[i])       # the "local solve", in closed form
            r[i] = u[i] - p.scratch[i] - 1.0
        end
        return nothing
    end
    function tiny_S!(S, u, p)
        fill!(S, 0.0)
        for i in axes(S, 1)
            S[i, i] = 0.4                          # deliberately not the exact tangent, so
        end                                        # the solve takes several iterations
        return nothing
    end
    function tiny_commit!(q_dest, u, p)
        for i in eachindex(q_dest)
            p.q[i] = 0.5 * tanh(u[i])
            q_dest[i] = p.q[i]
        end
        return true
    end

    n = 2
    f = MultiLevelNonlinearFunction(
        NonlinearFunction(tiny_R!; jac = tiny_S!, jac_prototype = zeros(n, n));
        primary = 1:n, internal = (n + 1):(2n), commit_internal! = tiny_commit!
    )
    prob = NonlinearProblem(f, zeros(2n), Tiny(zeros(n), zeros(n)))
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12, maxiters = 1000)

    step!(cache)
    step!(cache)                                   # warm up and compile
    @test NonlinearSolveBase.not_terminated(cache) # otherwise the measurement is of a no-op
    @test (@allocated step!(cache)) == 0
end
