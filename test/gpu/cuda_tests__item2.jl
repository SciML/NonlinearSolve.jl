using NonlinearSolve

using CUDA, NonlinearSolveBase, Test, LinearAlgebra

if CUDA.functional()
    CUDA.allowscalar(false)
    du = cu(rand(4))
    u = cu(rand(4))
    uprev = cu(rand(4))
    TERMINATION_CONDITIONS = [
        RelTerminationMode, AbsTerminationMode,
    ]
    NORM_TERMINATION_CONDITIONS = [
        AbsNormTerminationMode, RelNormTerminationMode, RelNormSafeTerminationMode,
        AbsNormSafeTerminationMode, RelNormSafeBestTerminationMode, AbsNormSafeBestTerminationMode,
    ]

    @testset begin
        @testset "Mode: $(tcond)" for tcond in TERMINATION_CONDITIONS
            @test_nowarn NonlinearSolveBase.check_convergence(
                tcond(), du, u, uprev, 1.0e-3, 1.0e-3
            )
        end

        @testset "Mode: $(tcond)" for tcond in NORM_TERMINATION_CONDITIONS
            for nfn in (
                    Base.Fix1(maximum, abs), Base.Fix2(norm, 2), Base.Fix2(norm, Inf),
                )
                @test_nowarn NonlinearSolveBase.check_convergence(
                    tcond(nfn), du, u, uprev, 1.0e-3, 1.0e-3
                )
            end
        end
    end
end
