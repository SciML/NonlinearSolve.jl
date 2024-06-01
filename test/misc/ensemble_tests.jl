@testitem "Ensemble Nonlinear Problems" tags=[:misc] begin
    using NonlinearSolve

    prob_func(prob, i, repeat) = remake(prob; u0 = prob.u0[:, i])

    prob_nls_oop = NonlinearProblem((u, p) -> u .* u .- p, rand(4, 128), 2.0)
    prob_nls_iip = NonlinearProblem((du, u, p) -> du .= u .* u .- p, rand(4, 128), 2.0)
    prob_nlls_oop = NonlinearLeastSquaresProblem((u, p) -> u .^ 2 .- p, rand(4, 128), 2.0)
    prob_nlls_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p; resid_prototype = rand(4)),
        rand(4, 128), 2.0)

    for prob in (prob_nls_oop, prob_nls_iip, prob_nlls_oop, prob_nlls_iip)
        ensembleprob = EnsembleProblem(prob; prob_func)

        for ensemblealg in (EnsembleThreads(), EnsembleSerial())
            sim = solve(ensembleprob, nothing, ensemblealg; trajectories = size(prob.u0, 2))
            @test all(SciMLBase.successful_retcode, sim.u)
        end
    end
end
