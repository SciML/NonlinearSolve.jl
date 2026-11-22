using NonlinearSolveBase: AbsNormSafeBestTerminationMode
using NonlinearSolveFirstOrder: NewtonRaphson
using SciMLBase: NonlinearProblem, ReturnCode, solve
using Test

params = (; root = fill(1.0e-4, 4), slope = 4.0e6)
residual(u, p) = expm1.(p.slope .* (u .- p.root))
abstol = 1.0e-6
weighted_norm(x) = 8 * maximum(abs, x)
mode = AbsNormSafeBestTerminationMode(weighted_norm; max_stalled_steps = 32)

@testset "$name" for (name, offset, options) in (
        ("Default", 27, (;)),
        ("Weighted maximum norm", 40, (; termination_condition = mode)),
    )
    # Newton steps start near 1/slope, so the step norm matters on the longer solve.
    prob = NonlinearProblem(residual, params.root .+ offset / params.slope, params)
    sol = solve(prob, NewtonRaphson(); abstol, options...)
    @test sol.retcode == ReturnCode.Success
    @test maximum(abs, sol.u .- params.root) <= 3.0e-13
end
