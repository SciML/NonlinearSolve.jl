using NonlinearSolveBase, SciMLBase, Test
using LinearAlgebra: norm

@testset "safe modes break on a non-finite objective" begin
    prob = NonlinearProblem((u, p) -> u, [1.0, 1.0])
    u = [1.0, 1.0]
    modes = (
        NonlinearSolveBase.AbsNormSafeTerminationMode,
        NonlinearSolveBase.AbsNormSafeBestTerminationMode,
        NonlinearSolveBase.RelNormSafeTerminationMode,
        NonlinearSolveBase.RelNormSafeBestTerminationMode,
    )
    @testset "$M / $nrm / du = $du" for M in modes,
            nrm in (Base.Fix2(norm, 2), Base.Fix1(maximum, abs)),
            du in ([Inf, 1.0], [NaN, 1.0], [-Inf, Inf])

        cache = SciMLBase.init(prob, M(nrm), copy(du), u)
        @test cache(du, u, u, 1.0e-8, 1.0e-8)
        @test cache.retcode == SciMLBase.ReturnCode.Unstable
    end
end

@testset "complex norms do not overflow" begin
    @test NonlinearSolveBase.L2_NORM(1.0e200 + 1.0e200im) ≈ sqrt(2) * 1.0e200
    @test NonlinearSolveBase.Linf_NORM(1.0e200 + 1.0e200im) ≈ sqrt(2) * 1.0e200
end
