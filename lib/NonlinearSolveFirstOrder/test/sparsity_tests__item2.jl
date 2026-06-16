using NonlinearSolveFirstOrder

using SparseConnectivityTracer, BandedMatrices, LinearAlgebra, SparseArrays,
    SparseMatrixColorings, ADTypes
using DifferentiationInterface: DenseSparsityDetector

N = 16
p = rand(N)
u0 = rand(N)

function f!(du, u, p)
    for i in 2:(length(u) - 1)
        du[i] = u[i - 1] - 2u[i] + u[i + 1] + p[i]
    end
    du[1] = -2u[1] + u[2] + p[1]
    du[end] = u[end - 1] - 2u[end] + p[end]
    return nothing
end

function f(u, p)
    du = similar(u, promote_type(eltype(u), eltype(p)))
    f!(du, u, p)
    return du
end

for nlf in (f, f!)
    @testset "Dense AD" begin
        nlprob = NonlinearProblem(NonlinearFunction(nlf), u0, p)

        cache = init(nlprob, NewtonRaphson(); abstol = 1.0e-9)
        @test cache.jac_cache.J isa Matrix
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Unstructured Sparse AD" begin
        nlprob_autosparse = NonlinearProblem(
            NonlinearFunction(nlf; sparsity = TracerSparsityDetector()),
            u0, p
        )

        cache = init(nlprob_autosparse, NewtonRaphson(); abstol = 1.0e-9)
        @test cache.jac_cache.J isa SparseMatrixCSC
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
    end

    # The detector runs its own ForwardDiff pass with a foreign tag; under the default
    # AutoSpecialize wrapping this used to throw `NoFunctionWrapperFoundError` because the
    # isbits foreign-tag duals matched no precompiled wrapper signature.
    @testset "Unstructured Sparse AD: DenseSparsityDetector" begin
        nlprob_dense_detector = NonlinearProblem(
            NonlinearFunction(
                nlf; sparsity = DenseSparsityDetector(AutoForwardDiff(); atol = 1.0e-4)
            ),
            u0, p
        )

        cache = init(nlprob_dense_detector, NewtonRaphson(); abstol = 1.0e-9)
        @test cache.jac_cache.J isa SparseMatrixCSC
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Structured Sparse AD: Banded Jacobian" begin
        jac_prototype = BandedMatrix(-1 => ones(N - 1), 0 => ones(N), 1 => ones(N - 1))
        nlprob_sparse_structured = NonlinearProblem(
            NonlinearFunction(nlf; jac_prototype), u0, p
        )

        cache = init(nlprob_sparse_structured, NewtonRaphson(); abstol = 1.0e-9)
        @test cache.jac_cache.J isa BandedMatrix
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Structured Sparse AD: Tridiagonal Jacobian" begin
        jac_prototype = Tridiagonal(ones(N - 1), ones(N), ones(N - 1))
        nlprob_sparse_structured = NonlinearProblem(
            NonlinearFunction(nlf; jac_prototype), u0, p
        )

        cache = init(nlprob_sparse_structured, NewtonRaphson(); abstol = 1.0e-9)
        @test cache.jac_cache.J isa Tridiagonal
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
    end
end
