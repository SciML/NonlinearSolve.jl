using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using ADTypes, SparseConnectivityTracer, SparseMatrixColorings

# Filter autodiff backends based on Julia version
autodiff_backends = [AutoForwardDiff(), AutoFiniteDiff(), AutoZygote()]
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    push!(autodiff_backends, AutoEnzyme())
end

@testset for ad in autodiff_backends
    @testset for u0 in ([1.0, 1.0], 1.0)
        prob = NonlinearProblem(
            NonlinearFunction(quadratic_f; sparsity = TracerSparsityDetector()), u0, 2.0
        )

        @testset "Newton Raphson" begin
            sol = solve(prob, NewtonRaphson(; autodiff = ad))
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1.0e-9
        end

        @testset "Trust Region" begin
            sol = solve(prob, TrustRegion(; autodiff = ad))
            err = maximum(abs, quadratic_f(sol.u, 2.0))
            @test err < 1.0e-9
        end
    end
end
