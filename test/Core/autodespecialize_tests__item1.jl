using NonlinearSolve, SciMLBase, Test
using SymbolicIndexingInterface: parameter_values

struct SolveDynamicParameters
    rate::Float64
end

struct SolveOtherDynamicParameters
    rate::Float64
    unused::Int
end

function solve_dynamic_residual!(resid, u, p)
    resid[1] = u[1]^2 - p.rate
    return nothing
end

function solve_dynamic_problem(p)
    f = NonlinearFunction{true, SciMLBase.AutoDespecialize}(solve_dynamic_residual!)
    return NonlinearProblem(f, [1.0], p)
end

first_prob = solve_dynamic_problem(SolveDynamicParameters(2.0))
second_prob = solve_dynamic_problem(SolveOtherDynamicParameters(3.0, 1))
first_cache = init(first_prob, NewtonRaphson())
second_cache = init(second_prob, NewtonRaphson())

@test typeof(first_cache) === typeof(second_cache)
@test first_cache.prob.p isa SciMLBase.DespecializedParameters
@test second_cache.prob.p isa SciMLBase.DespecializedParameters

first_sol = solve!(first_cache)
second_sol = solve!(second_cache)
@test SciMLBase.successful_retcode(first_sol)
@test SciMLBase.successful_retcode(second_sol)
@test first_sol.u[1] ≈ sqrt(2.0)
@test second_sol.u[1] ≈ sqrt(3.0)

replacement_parameters = SolveOtherDynamicParameters(4.0, 2)
for alg in (NewtonRaphson(), SimpleNewtonRaphson(), nothing)
    cache = init(first_prob, alg)
    reinit!(cache, [1.0]; p = replacement_parameters)
    @test parameter_values(cache) isa SciMLBase.DespecializedParameters
    @test SciMLBase.unwrap_parameters(parameter_values(cache)) === replacement_parameters
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0
end
