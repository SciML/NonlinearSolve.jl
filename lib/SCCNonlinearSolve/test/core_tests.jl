@testsetup module CoreRootfindTesting

include("../../../common/common_rootfind_testing.jl")

end

@testitem "Manual SCC" setup=[CoreRootfindTesting] tags=[:core] begin
    function f(du,u,p)
        du[1] = cos(u[2]) - u[1]
        du[2] = sin(u[1] + u[2]) + u[2]
        du[3] = 2u[4] + u[3] + 1.0
        du[4] = u[5]^2 + u[4]
        du[5] = u[3]^2 + u[5]
        du[6] = u[1] + u[2] + u[3] + u[4] + u[5]    + 2.0u[6] + 2.5u[7] + 1.5u[8]
        du[7] = u[1] + u[2] + u[3] + 2.0u[4] + u[5] + 4.0u[6] - 1.5u[7] + 1.5u[8]
        du[8] = u[1] + 2.0u[2] + 3.0u[3] + 5.0u[4] + 6.0u[5] + u[6] - u[7] - u[8]
    end
    prob = NonlinearProblem(f, zeros(8))
    sol = solve(prob)

    u0 = zeros(2)
    cache = zeros(3)

    function f1(du,u,(cache,p))
        du[1] = cos(u[2]) - u[1]
        du[2] = sin(u[1] + u[2]) + u[2]
    end
    explicitfun1(cache,sols) = nothing
    prob1 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), zeros(2),(cache,nothing))
    sol1 = solve(prob1, NewtonRaphson())

    function f2(du,u,(cache,p))
        du[1] = 2u[2] + u[1] + 1.0
        du[2] = u[3]^2 + u[2]
        du[3] = u[1]^2 + u[3]
    end
    explicitfun2(cache,sols) = nothing
    prob2 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), zeros(3),(cache,nothing))
    sol2 = solve(prob2, NewtonRaphson())

    function f3(du,u,(cache,p))
        du[1] = cache[1] + 2.0u[1] + 2.5u[2] + 1.5u[3]
        du[2] = cache[2] + 4.0u[1] - 1.5u[2] + 1.5u[3]
        du[3] = cache[3] + + u[1] - u[2] - u[3]
    end
    prob3 = NonlinearProblem(NonlinearFunction{true, SciMLBase.NoSpecialize}(f3), zeros(3),(cache,nothing))
    function explicitfun3(cache,sols)
        cache[1] = sols[1][1] + sols[1][2] + sols[2][1] + sols[2][2] + sols[2][3]
        cache[2] = sols[1][1] + sols[1][2] + sols[2][1] + 2.0sols[2][2] + sols[2][3]
        cache[3] = sols[1][1] + 2.0sols[1][2] + 3.0sols[2][1] + 5.0sols[2][2] + 6.0sols[2][3]
    end
    explicitfun3(cache,[sol1,sol2])
    sol3 = solve(prob3, NewtonRaphson())
    manualscc = [sol1; sol2; sol3]

    function SciMLBase.solve(prob::SciMLBase.SCCNonlinearProblem, alg; kwargs...)
        numscc = length(prob.probs)
        sols = [SciMLBase.build_solution(prob, nothing, prob.u0, convert(eltype(u0),NaN)*prob.u0) for prob in prob.probs]
        u = reduce(vcat,[prob.u0 for prob in prob.probs])
        resid = copy(u)

        for i in 1:numscc
            prob.explictfuns![i](prob.probs[i].p[1],sols)
            sol = solve(prob.probs[i], alg)
            _sol = SciMLBase.build_solution(prob.probs[i], nothing, sol.u, sol.resid, retcode = sol.retcode)
            sols[i] = _sol
        end
        
        # TODO: fix allocations with a lazy concatenation
        u .= reduce(vcat,sols)
        resid .= reduce(vcat,getproperty.(sols,:resid))

        SciMLBase.build_solution(prob, alg, u, resid, original = sols)
    end

    sccprob = SciMLBase.SCCNonlinearProblem([prob1,prob2,prob3], SciMLBase.Void{Any}.([explicitfun1,explicitfun2,explicitfun3]))
    scc_sol = solve(sccprob, NewtonRaphson())
    @test sol ≈ manualscc ≈ scc_sol

end