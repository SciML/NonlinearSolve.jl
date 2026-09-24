using NonlinearSolve
include("setup_wrappernllssetup.jl")

import LeastSquaresOptim

nlls_problems = [prob_oop, prob_iip]

solvers = []
for alg in (:lm, :dogleg),
        autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff(), :central, :forward)

    push!(solvers, LeastSquaresOptimJL(alg; autodiff))
end

for prob in nlls_problems, solver in solvers

    sol = solve(prob, solver; maxiters = 10000, abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.resid) < 1.0e-6
end
