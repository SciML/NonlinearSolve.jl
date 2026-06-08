using NonlinearSolve

using CUDA, NonlinearSolve, LinearSolve, StableRNGs, ADTypes

if CUDA.functional()
    CUDA.allowscalar(false)

    A = cu(rand(StableRNG(0), 4, 4))
    u0 = cu(rand(StableRNG(0), 4))
    b = cu(rand(StableRNG(0), 4))

    linear_f(du, u, p) = (du .= A * u .+ b)

    prob = NonlinearProblem(linear_f, u0)

    # ForwardDiff uses scalar indexing which doesn't work on GPU
    # Use AutoFiniteDiff for GPU-compatible Jacobian computation
    fd_autodiff = AutoFiniteDiff()

    SOLVERS = (
        NewtonRaphson(; autodiff = fd_autodiff),
        LevenbergMarquardt(; linsolve = QRFactorization(), autodiff = fd_autodiff),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES(), autodiff = fd_autodiff),
        PseudoTransient(; autodiff = fd_autodiff),
        Klement(),
        Broyden(; linesearch = LiFukushimaLineSearch()),
        LimitedMemoryBroyden(; threshold = 2, linesearch = LiFukushimaLineSearch()),
        DFSane(),
        TrustRegion(; linsolve = QRFactorization(), autodiff = fd_autodiff),
        TrustRegion(; linsolve = KrylovJL_GMRES(), concrete_jac = true, autodiff = fd_autodiff),  # Needed if Zygote not loaded
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
