using NonlinearSolveBase: NonlinearSolveBase, Utils
using ArrayInterface
using LinearSolve  # linsolve_identity!! routes matrices through the LinearSolve default
using LinearAlgebra
using Test

struct NoFastScalarMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end

Base.size(A::NoFastScalarMatrix) = size(A.data)
Base.getindex(A::NoFastScalarMatrix, i::Int, j::Int) = A.data[i, j]
Base.copy(A::NoFastScalarMatrix) = copy(A.data)
ArrayInterface.fast_scalar_indexing(::Type{<:NoFastScalarMatrix}) = false

@testset "nonsingular input matches inv" begin
    n = 20
    A_general = rand(n, n) + n * I
    A_triu = triu(rand(n, n) + n * I)
    A_tril = tril(rand(n, n) + n * I)

    for A in (A_general, A_triu, A_tril)
        A_orig = copy(A)
        workspace, A_ret = Utils.linsolve_workspace(A)
        @test A_ret === A
        Ai = Utils.linsolve_identity!!(workspace, A)
        @test Ai ≈ inv(A_orig)
        @test A == A_orig  # input must not be mutated
        # repeated call with the same workspace gives the same answer
        @test Utils.linsolve_identity!!(workspace, A) ≈ inv(A_orig)
        # nothing workspace (preinverted-init path in QuasiNewton) still works
        @test Utils.linsolve_identity!!(nothing, A) ≈ inv(A_orig)
        # re-inverting the returned inverse (which aliases the workspace's solution
        # buffer) recovers A
        Ai_copy = copy(Utils.linsolve_identity!!(workspace, A))
        @test Utils.linsolve_identity!!(workspace, Ai_copy) ≈ A_orig
        @test Utils.linsolve_identity!!(workspace, Utils.linsolve_identity!!(workspace, A)) ≈
            A_orig
    end
end

@testset "dense triangular input preserves triangular solve semantics" begin
    A = [1.1 0.0 0.0; 3.2 2.2 0.0; -4.1 5.3 3.3]
    expected = Matrix{Float64}(I, 3, 3)
    ldiv!(LowerTriangular(A), expected)
    workspace, _ = Utils.linsolve_workspace(A)
    @test Utils.linsolve_identity!!(workspace, A) == expected

    A_reset = [0.1 0.0 0.0; 100.3 0.2 0.0; -44.1 55.3 0.3]
    expected_reset = Matrix{Float64}(I, 3, 3)
    ldiv!(LowerTriangular(A_reset), expected_reset)
    @test Utils.linsolve_identity!!(workspace, A_reset) == expected_reset

    A_general = [2.0 1.0 0.0; 0.0 3.0 1.0; 1.0 0.0 4.0]
    copyto!(workspace.rhs, A_general)
    @test Utils.linsolve_identity!!(workspace, workspace.rhs) ≈ inv(A_general)
end

@testset "arrays without fast scalar indexing use pinv" begin
    A = NoFastScalarMatrix(rand(5, 5))
    workspace, A_ret = Utils.linsolve_workspace(A)
    @test workspace === nothing && A_ret === A
    @test Utils.linsolve_identity!!(workspace, A) ≈ pinv(A.data)
end

@testset "singular input takes the pivoted-QR rescue" begin
    # The result is the LinearSolve default algorithm's least-squares generalized
    # inverse from its singular-LU → pivoted-QR rescue, NOT the SVD `pinv` (an
    # intentional semantics change: same fitness for quasi-Newton initialization,
    # different finite matrix).
    n = 20
    A_singular = let B = rand(n, n)
        B[:, 1] .= 0
        B
    end
    A_singular_triu = triu(rand(n, n) + n * I)
    A_singular_triu[1, 1] = 0.0

    for A in (A_singular, A_singular_triu)
        A_orig = copy(A)
        workspace, _ = Utils.linsolve_workspace(A)
        X = Utils.linsolve_identity!!(workspace, A)
        @test size(X) == size(A)
        @test all(isfinite, X)
        # X is a least-squares generalized inverse: A * X projects onto range(A)
        @test A * X * A ≈ A atol = 1.0e-8 * norm(A)
        @test A == A_orig
        # a subsequent nonsingular solve with the same workspace is unaffected
        B = rand(n, n) + n * I
        @test Utils.linsolve_identity!!(workspace, B) ≈ inv(B)
    end
end

@testset "workspace reuse does not allocate a matrix copy" begin
    n = 51
    A = rand(n, n) + n * I
    workspace, _ = Utils.linsolve_workspace(A)
    Utils.linsolve_identity!!(workspace, A)  # compile
    # Steady state is the LinearSolve refactorization floor (`lu!` ipiv + wrapper); the
    # old code copied A (`lu`) and allocated a LAPACK getri work array, both O(n^2).
    allocs = @allocated Utils.linsolve_identity!!(workspace, A)
    @test allocs < sizeof(A)

    A_triu = triu(A)
    Utils.linsolve_identity!!(workspace, A_triu)  # compile
    allocs_tri = @allocated Utils.linsolve_identity!!(workspace, A_triu)
    @test allocs_tri < sizeof(A)
end

@testset "sparse matrices route through the lincache (no sparse pinv)" begin
    using SparseArrays
    m = 25
    As = spdiagm(
        -1 => fill(0.3, m - 1), 0 => collect(range(2.0, 3.0; length = m)),
        1 => fill(0.3, m - 1)
    )
    workspace, dense_buf = Utils.linsolve_workspace(As)
    @test dense_buf isa Matrix
    X = Utils.linsolve_identity!!(workspace, As)
    @test X isa Matrix && all(isfinite, X)
    @test X ≈ inv(Matrix(As)) rtol = 1.0e-10
    # nothing workspace (the preinverted-init path) also works with sparse input
    X2 = Utils.linsolve_identity!!(nothing, As)
    @test X2 ≈ X rtol = 1.0e-10
    # repeated call with a fresh sparse A stays cheap (copyto! into the dense buffer,
    # no sparse svd/pinv path)
    Utils.linsolve_identity!!(workspace, As)
    allocs = @allocated Utils.linsolve_identity!!(workspace, As)
    @test allocs < sizeof(dense_buf)
end

@testset "structured input takes the lincache with dense buffers" begin
    T = Tridiagonal(fill(0.3, 4), collect(range(2.0, 3.0; length = 5)), fill(0.3, 4))
    workspace, T_ret = Utils.linsolve_workspace(T)
    @test T_ret === T
    X = Utils.linsolve_identity!!(workspace, T)
    @test X ≈ inv(Matrix(T))
    @test Utils.linsolve_identity!!(nothing, T) ≈ inv(Matrix(T))
end

@testset "native fast paths: Number / Diagonal / SMatrix" begin
    using StaticArrays

    # scalars: zero-guarded inverse, identical values to the old `pinv(x::Number)`
    @test Utils.linsolve_identity!!(nothing, 2.0) == 0.5
    @test Utils.linsolve_identity!!(nothing, 0.0) == 0.0
    @test Utils.linsolve_identity!!(nothing, 0) == 0.0
    @test Utils.linsolve_workspace(2.0) === (nothing, 2.0)

    # Diagonal: entrywise zero-guarded inverse, no lincache workspace
    D = Diagonal([2.0, 0.0, 4.0])
    ws_D, D_ret = Utils.linsolve_workspace(D)
    @test ws_D === nothing && D_ret === D
    Di = Utils.linsolve_identity!!(ws_D, D)
    @test Di isa Diagonal && Di.diag == [0.5, 0.0, 0.25]

    # SMatrix: stays on the allocation-free native `pinv` (quasi-Newton true-jacobian
    # init with SArray states reaches this with a `nothing` workspace, and a singular
    # initial Jacobian must stay finite where the native `\` would error)
    As = SA[2.0 0.0; 1.0 3.0]
    ws_S, As_ret = Utils.linsolve_workspace(As)
    @test ws_S === nothing && As_ret === As
    Xs = Utils.linsolve_identity!!(ws_S, As)
    @test Xs isa SMatrix && Xs ≈ inv(As)
    As_sing = SA[0.0 0.0; 0.0 0.0]
    Xs_sing = Utils.linsolve_identity!!(nothing, As_sing)
    @test Xs_sing isa SMatrix && all(iszero, Xs_sing)
end
