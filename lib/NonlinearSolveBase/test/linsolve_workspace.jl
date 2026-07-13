using NonlinearSolveBase: NonlinearSolveBase, Utils
using LinearSolve  # used by sparse and structured linsolve_identity!! workspaces
using LinearAlgebra
using Test

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
        # re-inverting the returned inverse recovers A
        Ai_copy = copy(Utils.linsolve_identity!!(workspace, A))
        @test Utils.linsolve_identity!!(workspace, Ai_copy) ≈ A_orig
        @test Utils.linsolve_identity!!(workspace, Utils.linsolve_identity!!(workspace, A)) ≈
            A_orig
    end
end

@testset "dense singular input matches pinv" begin
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
        @test workspace === nothing
        X = Utils.linsolve_identity!!(workspace, A)
        @test size(X) == size(A)
        @test all(isfinite, X)
        @test X ≈ pinv(A_orig)
        @test A * X * A ≈ A atol = 1.0e-8 * norm(A)
        @test A == A_orig
        B = rand(n, n) + n * I
        @test Utils.linsolve_identity!!(workspace, B) ≈ inv(B)
    end
end

@testset "strided matrices use the native inverse path" begin
    n = 51
    A = rand(n, n) + n * I
    workspace, A_ret = Utils.linsolve_workspace(A)
    @test workspace === nothing && A_ret === A
    @test Utils.linsolve_identity!!(workspace, A) ≈ inv(A)

    A_triu = triu(A)
    workspace, A_triu_ret = Utils.linsolve_workspace(A_triu)
    @test workspace === nothing && A_triu_ret === A_triu
    @test Utils.linsolve_identity!!(workspace, A_triu) ≈ inv(A_triu)
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
