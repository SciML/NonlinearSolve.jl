using NonlinearSolveBase, LinearSolve, LinearAlgebra, SciMLBase

# Dense `LUFactorization` refactorization through `construct_linear_solver` must not
# copy `A`: the cache is inited on a NonlinearSolve-owned copy with `alias_A = true`,
# so LinearSolve refactorizes in place (`lu!`) and `set_lincache_A!` refreshes the
# aliased buffer via `copyto!` before every refactorization.

@test NonlinearSolveBase.alias_A_for_refactorization(LUFactorization(), rand(4, 4))
@test NonlinearSolveBase.alias_A_for_refactorization(nothing, rand(4, 4))
@test !NonlinearSolveBase.alias_A_for_refactorization(QRFactorization(), rand(4, 4))
@test !NonlinearSolveBase.alias_A_for_refactorization(nothing, Diagonal(rand(4)))

refactorize_and_solve!(lc, A, b, u) = lc(; A, b, linu = u)

n = 50
A = rand(n, n) + 5I
b = rand(n)
u = zeros(n)
stats = SciMLBase.NLStats(0, 0, 0, 0, 0)

lc = NonlinearSolveBase.construct_linear_solver(
    nothing, LUFactorization(), A, b, u, nothing; stats
)
@test lc isa NonlinearSolveBase.LinearSolveJLCache

# The first factorization must not destroy the caller's `A` (consumers like
# trust-region and line-search methods read `J` after the solve).
A_snapshot = copy(A)
res = refactorize_and_solve!(lc, A, b, u)
@test A == A_snapshot
@test res.u ≈ A \ b

# Refactorization with an updated `A` stays correct (the aliased buffer was destroyed
# by `lu!`, so this exercises the full `copyto!` refresh).
A2 = rand(n, n) + 5I
res = refactorize_and_solve!(lc, A2, b, u)
@test A2 != A_snapshot && res.u ≈ A2 \ b
@test stats.nfactors == 2

# Allocation bound: the O(n²) per-refactorization copy (`lu` instead of `lu!`) is
# `sizeof(A)` = 20 kB here; without it only the per-call `ipiv` inside
# `LinearAlgebra.lu!` remains (~0.5 kB). The in-place refactorization path gated on
# `alias_A` only exists in LinearSolve ≥ 4.2 (older versions always `lu`-copy; the
# aliased init is still correct there, just not allocation-free), so only assert the
# bound where the path exists — e.g. downgrade CI resolves LinearSolve 3.x.
if pkgversion(LinearSolve) >= v"4.2.0"
    allocs = @allocated refactorize_and_solve!(lc, A, b, u)
    @test allocs < sizeof(A) ÷ 2
end

# `linsolve = nothing` (the standard default) resolves to `DefaultLinearSolver`; its
# dense LU choice funnels through the same alias-gated body, and its singular-LU → QR
# safety fallback keeps a cached private backup, so refactorization stays cheap and
# the caller's `A` is still never destroyed.
lc_def = NonlinearSolveBase.construct_linear_solver(
    nothing, nothing, A, b, u, nothing; stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
)
A_snapshot = copy(A)
res = refactorize_and_solve!(lc_def, A, b, u)
@test A == A_snapshot
@test res.u ≈ A \ b
res = refactorize_and_solve!(lc_def, A2, b, u)
@test res.u ≈ A2 \ b
if pkgversion(LinearSolve) >= v"4.2.0"
    refactorize_and_solve!(lc_def, A, b, u)
    allocs = @allocated refactorize_and_solve!(lc_def, A2, b, u)
    @test allocs < sizeof(A) ÷ 2
end
