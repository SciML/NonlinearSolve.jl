using NonlinearSolveFirstOrder, LinearSolve, SciMLOperators, SciMLBase
using Enzyme # Activate the preferred reverse-mode backend before checking cache inference.
using LinearAlgebra, SparseArrays, Test
using SciMLOperators: AbstractSciMLOperator, isconvertible

# A `jac_prototype` that is an `AbstractSciMLOperator` is handed to the solver as the
# Jacobian directly: an iterative solver applies it matrix-free via `mul!`, while a
# factorization materializes it via `convert(AbstractMatrix, ·)`. The choice is
# routed by `needs_concrete_A(linsolve)` (like NLNewton); `isconvertible(op)` guards the
# factorization path.

const N = 40
const Wmat = sparse(Tridiagonal(fill(-1.0, N - 1), fill(4.0, N), fill(-1.0, N - 1)))
const bvec = collect(1.0:N)
resid!(F, z, p) = (mul!(F, Wmat, z); F .-= bvec; return nothing)
const xref = Wmat \ bvec

@testset "convertible operator (MatrixOperator): matrix-free Krylov, materialized direct" begin
    mop = MatrixOperator(copy(Wmat))
    @test isconvertible(mop)
    prob = NonlinearProblem(NonlinearFunction(resid!; jac_prototype = mop), zeros(N))

    cache = @inferred init(prob, TrustRegion())
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ xref

    for ls in (KrylovJL_GMRES(), LUFactorization(), KLUFactorization())
        cache = init(prob, NewtonRaphson(linsolve = ls))
        # The cache holds the operator itself, never a residual-derived JacobianOperator.
        @test cache.jac_cache.J isa AbstractSciMLOperator
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ xref
    end
end

@testset "non-convertible operator: Krylov ok, factorization guarded" begin
    fop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Wmat, v), zeros(N), zeros(N);
        islinear = true
    )
    @test !isconvertible(fop)
    prob = NonlinearProblem(NonlinearFunction(resid!; jac_prototype = fop), zeros(N))

    # A concrete-A solver on a non-convertible operator errors clearly at construction
    # (rather than deep in LinearSolve's `convert`).
    @test_throws ArgumentError init(prob, NewtonRaphson(linsolve = LUFactorization()))
end

@testset "operator-free problems are unaffected" begin
    f!(F, z, p) = (@. F = z^2 - 2; nothing)
    sol = solve(NonlinearProblem(NonlinearFunction(f!), ones(3)), NewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    @test all(x -> abs(x - sqrt(2)) < 1.0e-8, sol.u)
end
