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
