using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

prob = NonlinearProblem(quadratic_f, ones(4), 2.0; maxiters = 2)
sol = solve(prob, SimpleNewtonRaphson())
@test sol.retcode === ReturnCode.MaxIters
