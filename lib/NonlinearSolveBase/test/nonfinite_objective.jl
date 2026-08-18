using NonlinearSolveBase, SciMLBase, Test
using LinearAlgebra: norm

# `L2_NORM` used to be written with `@fastmath`, whose `nnan`/`ninf` LLVM flags let the
# optimizer delete the `!isfinite(objective)` protective break in the safe termination
# modes (it compiled down to a constant `false`). A diverged solve then fell through to the
# ordinary termination logic. The fold only happens where the norm inlines into the code
# holding the check, so this has to be driven through the termination cache: calling
# `isfinite(L2_NORM(x))` from a `@test` does not reproduce it. Both norms are covered
# because only the `Base.Fix2(norm, 2)` -> `L2_NORM` path was affected.
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

# `@fastmath abs` on a `Complex` skips the `hypot` scaling and overflows.
@testset "complex norms do not overflow" begin
    @test NonlinearSolveBase.L2_NORM(1.0e200 + 1.0e200im) ≈ sqrt(2) * 1.0e200
    @test NonlinearSolveBase.Linf_NORM(1.0e200 + 1.0e200im) ≈ sqrt(2) * 1.0e200
end
