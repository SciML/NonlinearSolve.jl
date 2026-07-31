using NonlinearSolve
using SciMLBase
import NonlinearSolveBase

step_drive!(cache) = while NonlinearSolveBase.not_terminated(cache)
    SciMLBase.step!(cache)
end

f(u, p) = u .^ 2 .- p

@testset "step!-driven polyalgorithm cache survives reinit! ($(nameof(typeof(alg))))" for alg in (
        RobustMultiNewton(), FastShortcutNonlinearPolyalg(),
    )
    prob = NonlinearProblem(f, [1.0], 4.0)
    cache = init(prob, alg)

    step_drive!(cache)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test NonlinearSolveBase.get_u(cache) ≈ [2.0] atol = 1.0e-6

    SciMLBase.reinit!(cache, [1.0]; p = 9.0)
    @test NonlinearSolveBase.not_terminated(cache)
    @test cache.retcode == SciMLBase.ReturnCode.Default
    step_drive!(cache)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test NonlinearSolveBase.get_u(cache) ≈ [3.0] atol = 1.0e-6
    @test NonlinearSolveBase.get_fu(cache) ≈ [0.0] atol = 1.0e-6
end
