using NonlinearSolveFirstOrder
include("setup_corenllstesting.jl")

using LinearAlgebra

nlls_problems = [prob_oop, prob_iip, prob_oop_vjp, prob_iip_vjp]

for prob in nlls_problems, solver in solvers

    sol = solve(prob, solver; maxiters = 10000, abstol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid, 2) < 1.0e-6
end
