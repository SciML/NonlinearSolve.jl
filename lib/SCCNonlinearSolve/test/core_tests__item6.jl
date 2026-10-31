using NonlinearSolve, SCCNonlinearSolve, SciMLBase, Test

struct SCCDynamicParameters
    target::Float64
end

struct SCCOtherDynamicParameters
    target::Float64
    unused::Int
end

function scc_dynamic_residual!(resid, u, p)
    resid[1] = u[1]^2 - p.target
    return nothing
end

const seen_scc_parameter = Ref{DataType}()

function scc_explicit!(p::Union{SCCDynamicParameters, SCCOtherDynamicParameters}, sols)
    seen_scc_parameter[] = typeof(p)
    return nothing
end

function scc_dynamic_problem(p, container)
    f = NonlinearFunction{true, SciMLBase.AutoDespecialize}(scc_dynamic_residual!)
    subprob = NonlinearProblem(f, [1.0], p)
    probs, explicitfuns = if container === :tuple
        ((subprob,), (scc_explicit!,))
    else
        ([subprob], [scc_explicit!])
    end
    return SciMLBase.SCCNonlinearProblem(probs, explicitfuns)
end

alg = SCCNonlinearSolve.SCCAlg(; nlalg = NewtonRaphson(), store_original = Val(true))
for container in (:tuple, :vector)
    @testset "$container storage" begin
        first_sol = solve(scc_dynamic_problem(SCCDynamicParameters(2.0), container), alg)
        @test seen_scc_parameter[] === SCCDynamicParameters
        second_sol = solve(
            scc_dynamic_problem(SCCOtherDynamicParameters(3.0, 1), container), alg
        )
        @test seen_scc_parameter[] === SCCOtherDynamicParameters

        @test SciMLBase.successful_retcode(first_sol)
        @test SciMLBase.successful_retcode(second_sol)
        @test first_sol.u[1] ≈ sqrt(2.0)
        @test second_sol.u[1] ≈ sqrt(3.0)
        @test typeof(first_sol.prob) === typeof(second_sol.prob)
        @test first_sol.prob.probs[1].p isa SciMLBase.DespecializedParameters
        @test second_sol.prob.probs[1].p isa SciMLBase.DespecializedParameters
        @test typeof(first_sol.original[1].prob) === typeof(second_sol.original[1].prob)
    end
end

@testset "AutoSpecialize remains unconcretized at the SCC boundary" begin
    f = NonlinearFunction{true, SciMLBase.AutoSpecialize}(scc_dynamic_residual!)
    subproblem = NonlinearProblem(f, [1.0], SCCDynamicParameters(2.0))
    problem = SciMLBase.SCCNonlinearProblem((subproblem,), (scc_explicit!,))
    solution = solve(problem, alg)
    @test solution.prob.probs[1].f.f === scc_dynamic_residual!
end
