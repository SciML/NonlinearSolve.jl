using NonlinearSolve
using LinearAlgebra

f_oop(u, p) = u .* u .- p

function hard_problem(u, p)
    return @. u^10 - p
end

function cubic(u, p)
    return u .^ 3 .- p
end

u0 = [1.0, 1.0]
prob = NonlinearProblem(f_oop, u0, 2.0)
simple_polyalg = NonlinearSolvePolyAlgorithm((SimpleNewtonRaphson(), SimpleBroyden()))
mixed_polyalg = NonlinearSolvePolyAlgorithm((SimpleNewtonRaphson(), NewtonRaphson()))

direct = solve(prob, simple_polyalg)
@test SciMLBase.successful_retcode(direct)

cache = init(prob, simple_polyalg)
sol = solve!(cache)
@test SciMLBase.successful_retcode(sol)
@test sol.u ≈ fill(√2, 2) atol = 1.0e-8
@test norm(f_oop(sol.u, 2.0), Inf) ≤ 1.0e-8
@test sol.resid ≈ f_oop(sol.u, 2.0) atol = 1.0e-12

reinit!(cache, u0)
sol_re = solve!(cache)
@test SciMLBase.successful_retcode(sol_re)
@test sol_re.u ≈ fill(√2, 2) atol = 1.0e-8
@test sol_re.resid ≈ f_oop(sol_re.u, 2.0) atol = 1.0e-12

alias_cache = init(prob, simple_polyalg; alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))
sol_alias = solve!(alias_cache)
@test SciMLBase.successful_retcode(sol_alias)
@test sol_alias.u ≈ fill(√2, 2) atol = 1.0e-8
@test sol_alias.resid ≈ f_oop(sol_alias.u, 2.0) atol = 1.0e-12

mixed_cache = init(prob, mixed_polyalg)
sol_mixed = solve!(mixed_cache)
@test SciMLBase.successful_retcode(sol_mixed)
@test sol_mixed.u ≈ fill(√2, 2) atol = 1.0e-8
@test sol_mixed.resid ≈ f_oop(sol_mixed.u, 2.0) atol = 1.0e-12

# Immediate Simple* win must keep the winning sub-solution's own trace (`nothing`),
# not a sibling stepping cache's unused history.
immediate_alg = NonlinearSolvePolyAlgorithm(
    (SimpleNewtonRaphson(), NewtonRaphson()); store_original = Val(true)
)
immediate_cache = init(
    NonlinearProblem(f_oop, [1.0], 2.0), immediate_alg; store_trace = Val(true)
)
sol_immediate = solve!(immediate_cache)
@test SciMLBase.successful_retcode(sol_immediate)
@test immediate_cache.best == 1
@test sol_immediate.trace === sol_immediate.original.trace
@test sol_immediate.trace === nothing
@test sol_immediate.resid ≈ f_oop(sol_immediate.u, 2.0) atol = 1.0e-12
@test maximum(abs, sol_immediate.resid) < 1.0e-12

# Simple* win after a stepping method fails must not attach the failed stepping history.
escalation_alg = NonlinearSolvePolyAlgorithm(
    (NewtonRaphson(), SimpleBroyden()); store_original = Val(true)
)
escalation_cache = init(
    NonlinearProblem(cubic, [0.0], 2.0), escalation_alg; store_trace = Val(true)
)
sol_escalation = solve!(escalation_cache)
@test SciMLBase.successful_retcode(sol_escalation)
@test escalation_cache.best == 2
@test sol_escalation.trace === sol_escalation.original.trace
@test sol_escalation.trace === nothing
@test sol_escalation.resid ≈ cubic(sol_escalation.u, 2.0) atol = 1.0e-12
@test maximum(abs, sol_escalation.resid) < 1.0e-10

hard = NonlinearProblem(hard_problem, [100.0, 200.0], [1.0, 1.0])
fallback_algs = (SimpleNewtonRaphson(), SimpleBroyden())
fallback_expected = let
    sols = map(a -> solve(remake(hard; u0 = copy(hard.u0)), a; maxiters = 1), fallback_algs)
    sols[argmin(map(s -> norm(s.resid, Inf), sols))]
end
fallback_cache = init(hard, NonlinearSolvePolyAlgorithm(fallback_algs); maxiters = 1)
sol_fb = solve!(fallback_cache)
@test sol_fb.u ≈ fallback_expected.u
@test sol_fb.resid ≈ hard_problem(sol_fb.u, hard.p)
@test sol_fb.retcode == fallback_expected.retcode

# Parameter-changing retain_best reinit must not return the previous solve's stepping
# history when a no-init method wins again.
retain_alg = NonlinearSolvePolyAlgorithm(
    (NewtonRaphson(), SimpleBroyden()); store_original = Val(true)
)
retain_cache = init(
    NonlinearProblem(cubic, [0.0], 2.0), retain_alg; store_trace = Val(true)
)
sol_retain_first = solve!(retain_cache)
@test SciMLBase.successful_retcode(sol_retain_first)
@test retain_cache.best == 2
old_history = copy(retain_cache.caches[1].trace.history)
reinit!(retain_cache, [2.5]; p = 27.0, retain_best = true)
sol_retain_second = solve!(retain_cache)
@test SciMLBase.successful_retcode(sol_retain_second)
@test retain_cache.best == 2
@test sol_retain_second.u ≈ [cbrt(27)] atol = 1.0e-8
@test sol_retain_second.trace === sol_retain_second.original.trace
@test sol_retain_second.trace === nothing
@test !(sol_retain_second.trace === retain_cache.caches[1].trace)
@test !(sol_retain_second.trace !== nothing && sol_retain_second.trace.history == old_history)

# Warmed IIP polyalgorithm solve! allocations must match master on ordinary
# (non-Simple) ladders. Values for Julia 1.10 are the minima from round-2
# `probe.jl` on master `06b12295` (12 warmed `@allocated solve!` calls with
# `reinit!` outside the measurement). Julia 1.12 master is allocation-free.
function _polyalg_warmed_iip_allocs(alg; n::Int = 12)
    f_iip!(du, u, p) = (du .= u .* u .- p)
    prob_iip = NonlinearProblem(f_iip!, [1.0, 1.0], 2.0)
    cache = alg === nothing ? init(prob_iip) : init(prob_iip, alg)
    solve!(cache)
    bytes = Vector{Int}(undef, n)
    for i in 1:n
        reinit!(cache, copy(prob_iip.u0); p = prob_iip.p)
        bytes[i] = @allocated solve!(cache)
    end
    return minimum(bytes)
end

@testset "ordinary IIP polyalgorithm solve! allocations" begin
    # master `06b12295` Julia 1.10.12 IIP success minima from round-2 probe.jl
    cases = (
        (nothing, 624),
        (FastShortcutNonlinearPolyalg(), 704),
        (RobustMultiNewton(), 896),
        (NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden())), 160),
    )
    for (alg, master_bytes_1_10) in cases
        alloc = _polyalg_warmed_iip_allocs(alg)
        if VERSION ≥ v"1.12"
            @test alloc == 0
        else
            @test alloc ≤ master_bytes_1_10
        end
    end
end
