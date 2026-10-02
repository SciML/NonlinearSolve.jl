using NonlinearSolve
using LinearAlgebra

f_oop(u, p) = u .* u .- p

function hard_problem(u, p)
    return @. u^10 - p
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

reinit!(cache, u0)
sol_re = solve!(cache)
@test SciMLBase.successful_retcode(sol_re)
@test sol_re.u ≈ fill(√2, 2) atol = 1.0e-8

alias_cache = init(prob, simple_polyalg; alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true))
sol_alias = solve!(alias_cache)
@test SciMLBase.successful_retcode(sol_alias)
@test sol_alias.u ≈ fill(√2, 2) atol = 1.0e-8

mixed_cache = init(prob, mixed_polyalg)
sol_mixed = solve!(mixed_cache)
@test SciMLBase.successful_retcode(sol_mixed)
@test sol_mixed.u ≈ fill(√2, 2) atol = 1.0e-8

hard = NonlinearProblem(hard_problem, [100.0, 200.0], [1.0, 1.0])
fallback_cache = init(hard, simple_polyalg; maxiters = 1)
sol_fb = solve!(fallback_cache)
@test sol_fb.u isa AbstractVector
@test sol_fb.resid isa AbstractVector
@test SciMLBase.successful_retcode(sol_fb) ||
    sol_fb.retcode ∈ (ReturnCode.MaxIters, ReturnCode.Failure, ReturnCode.ConvergenceFailure)
