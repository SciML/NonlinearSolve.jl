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
            for alg in SOLVERS
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end

        linear_f(u, p) = A * u .+ b

        prob = NonlinearProblem{false}(linear_f, u0)

        @testset "[OOP] GPU Solvers" begin
            for alg in SOLVERS
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end
    end
end
