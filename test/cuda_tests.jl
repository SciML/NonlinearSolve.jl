@testitem "CUDA Tests" tags=[:cuda] begin
    using CUDA, NonlinearSolve, LinearSolve, StableRNGs

    if CUDA.functional()
        CUDA.allowscalar(false)

        A = cu(rand(StableRNG(0), 4, 4))
        u0 = cu(rand(StableRNG(0), 4))
        b = cu(rand(StableRNG(0), 4))

        linear_f(du, u, p) = (du .= A * u .+ b)

        prob = NonlinearProblem(linear_f, u0)

        SOLVERS = (
            NewtonRaphson(),
            LevenbergMarquardt(; linsolve = QRFactorization()),
            LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
            PseudoTransient(),
            Klement(),
            Broyden(; linesearch = LiFukushimaLineSearch()),
            LimitedMemoryBroyden(; threshold = 2, linesearch = LiFukushimaLineSearch()),
            DFSane(),
            TrustRegion(; linsolve = QRFactorization()),
            TrustRegion(; linsolve = KrylovJL_GMRES(), concrete_jac = true),  # Needed if Zygote not loaded
            nothing
        )

        @testset "[IIP] GPU Solvers" begin
            @testset "$(nameof(typeof(alg)))" for alg in SOLVERS
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end

        linear_f(u, p) = A * u .+ b

        prob = NonlinearProblem{false}(linear_f, u0)

        @testset "[OOP] GPU Solvers" begin
            @testset "$(nameof(typeof(alg)))" for alg in SOLVERS
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end
    end
end

@testitem "Termination Conditions: Allocations" tags=[:cuda] begin
    using CUDA, NonlinearSolveBase, Test, LinearAlgebra
    CUDA.allowscalar(false)
    du = cu(rand(4))
    u = cu(rand(4))
    uprev = cu(rand(4))
    TERMINATION_CONDITIONS = [
        RelTerminationMode, AbsTerminationMode
    ]
    NORM_TERMINATION_CONDITIONS = [
        AbsTerminationMode, AbsNormTerminationMode, RelNormTerminationMode, RelNormSafeTerminationMode,
        AbsNormSafeTerminationMode, RelNormSafeBestTerminationMode, AbsNormSafeBestTerminationMode
    ]

    @testset begin
        @testset "Mode: $(tcond)" for tcond in TERMINATION_CONDITIONS
            @test_nowarn NonlinearSolveBase.check_convergence(
                tcond, du, u, uprev, 1e-3, 1e-3)
        end

        @testset "Mode: $(tcond)" for tcond in NORM_TERMINATION_CONDITIONS
            for nfn in (Base.Fix1(maximum, abs), Base.Fix2(norm, 2), Base.Fix2(norm, Inf))
                @test_nowarn NonlinearSolveBase.check_convergence(
                    tcond(nfn), du, u, uprev, 1e-3, 1e-3)
            end
        end
    end
end
