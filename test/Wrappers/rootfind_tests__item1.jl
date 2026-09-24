using NonlinearSolve

import NLSolvers, NLsolve, SIAMFANLEquations, MINPACK

function f_iip(du, u, p, t)
    du[1] = 2 - 2u[1]
    return du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob_iip = SteadyStateProblem(f_iip, u0)

algs = Any[
    NLSolversJL(NLSolvers.LineSearch(NLSolvers.Newton(), NLSolvers.Backtracking())),
    NLsolveJL(),
    SIAMFANLEquationsJL(),
    CMINPACK(),
]

@testset "$(nameof(typeof(alg)))" for alg in algs
    alg isa CMINPACK && Sys.isapple() && continue
    sol = solve(prob_iip, alg)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6
end

f_oop(u, p, t) = [2 - 2u[1], u[1] - 4u[2]]
u0 = zeros(2)
prob_oop = SteadyStateProblem(f_oop, u0)

@testset "$(nameof(typeof(alg)))" for alg in algs
    alg isa CMINPACK && Sys.isapple() && continue
    sol = solve(prob_oop, alg)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6
end
