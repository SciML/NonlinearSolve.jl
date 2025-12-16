@testitem "CUDA Tests" tags=[:cuda] skip=:(!isempty(VERSION.prerelease)) begin
    using CUDA, NonlinearSolve, LinearSolve, StableRNGs

    if CUDA.functional()
        CUDA.allowscalar(false)

        A=cu(rand(StableRNG(0), 4, 4))
        u0=cu(rand(StableRNG(0), 4))
        b=cu(rand(StableRNG(0), 4))

        linear_f(du, u, p)=(du.=A*u .+ b)

        prob=NonlinearProblem(linear_f, u0)

        SOLVERS=(
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

        linear_f(u, p)=A*u .+ b

        prob=NonlinearProblem{false}(linear_f, u0)

        @testset "[OOP] GPU Solvers" begin
            @testset "$(nameof(typeof(alg)))" for alg in SOLVERS
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end
    end
end

# Test for non-square Jacobian least squares problems on GPU
# See https://github.com/SciML/NonlinearSolve.jl/issues/746
@testitem "CUDA Least Squares Non-Square Jacobian" tags=[:cuda] skip=:(!isempty(VERSION.prerelease)) begin
    using CUDA, NonlinearSolve, LinearSolve, StableRNGs

    if CUDA.functional()
        # Allow scalar operations for this test since we're testing solver logic,
        # not GPU kernel performance
        CUDA.allowscalar(true)

        # Test case: residual size (4) != parameter size (2)
        N=4
        u0=cu(Float32[1.0, 1.0])
        p=cu(ones(Float32, N))
        resid_prototype=cu(zeros(Float32, N))

        function my_residuals!(du, u, p)
            du[1]=u[1]^2-p[1]
            du[2]=u[2]^2-p[2]
            du[3]=u[1]^2-p[3]
            du[4]=u[2]^2-p[4]
            return nothing
        end

        # Test with explicit Jacobian
        jac_prototype=cu(zeros(Float32, N, 2))

        function my_jac!(jac, u, p)
            jac[1, 1]=2*u[1]
            jac[1, 2]=0
            jac[2, 1]=0
            jac[2, 2]=2*u[2]
            jac[3, 1]=2*u[1]
            jac[3, 2]=0
            jac[4, 1]=0
            jac[4, 2]=2*u[2]
            return nothing
        end

        LS_SOLVERS=(
            GaussNewton(),
            LevenbergMarquardt()
        )

        @testset "[IIP] Least Squares Non-Square Jacobian" begin
            # Test with resid_prototype only (auto-diff Jacobian)
            prob_auto = NonlinearLeastSquaresProblem(
                NonlinearFunction(my_residuals!; resid_prototype = resid_prototype),
                u0, p
            )
            @testset "$(nameof(typeof(alg))) auto-diff" for alg in LS_SOLVERS
                @test_nowarn sol = solve(prob_auto, alg; abstol = 1.0f-4, reltol = 1.0f-4)
            end

            # Test with explicit Jacobian
            prob_explicit = NonlinearLeastSquaresProblem(
                NonlinearFunction(my_residuals!;
                    resid_prototype = resid_prototype,
                    jac = my_jac!,
                    jac_prototype = jac_prototype),
                u0, p
            )
            @testset "$(nameof(typeof(alg))) explicit jac" for alg in LS_SOLVERS
                @test_nowarn sol = solve(prob_explicit, alg; abstol = 1.0f-4, reltol = 1.0f-4)
            end
        end

        # Out-of-place version
        function my_residuals(u, p)
            T=eltype(u)
            return CuArray(T[
                u[1] ^ 2 - p[1], u[2] ^ 2 - p[2], u[1] ^ 2 - p[3], u[2] ^ 2 - p[4]])
        end

        @testset "[OOP] Least Squares Non-Square Jacobian" begin
            prob_oop = NonlinearLeastSquaresProblem{false}(
                NonlinearFunction{false}(my_residuals; resid_prototype = resid_prototype),
                u0, p
            )
            @testset "$(nameof(typeof(alg)))" for alg in LS_SOLVERS
                @test_nowarn sol = solve(prob_oop, alg; abstol = 1.0f-4, reltol = 1.0f-4)
            end
        end

        # Restore default scalar behavior
        CUDA.allowscalar(false)
    end
end

@testitem "Termination Conditions: Allocations" tags=[:cuda] skip=:(!isempty(VERSION.prerelease)) begin
    using CUDA, NonlinearSolveBase, Test, LinearAlgebra

    if CUDA.functional()
        CUDA.allowscalar(false)
        du=cu(rand(4))
        u=cu(rand(4))
        uprev=cu(rand(4))
        TERMINATION_CONDITIONS=[
            RelTerminationMode, AbsTerminationMode
        ]
        NORM_TERMINATION_CONDITIONS=[
            AbsNormTerminationMode, RelNormTerminationMode, RelNormSafeTerminationMode,
            AbsNormSafeTerminationMode, RelNormSafeBestTerminationMode, AbsNormSafeBestTerminationMode
        ]

        @testset begin
            @testset "Mode: $(tcond)" for tcond in TERMINATION_CONDITIONS
                @test_nowarn NonlinearSolveBase.check_convergence(
                    tcond(), du, u, uprev, 1e-3, 1e-3)
            end

            @testset "Mode: $(tcond)" for tcond in NORM_TERMINATION_CONDITIONS
                for nfn in (
                    Base.Fix1(maximum, abs), Base.Fix2(norm, 2), Base.Fix2(norm, Inf)
                )
                    @test_nowarn NonlinearSolveBase.check_convergence(
                        tcond(nfn), du, u, uprev, 1e-3, 1e-3)
                end
            end
        end
    end
end
