using NonlinearSolveBase, LinearSolve, LinearAlgebra, StaticArrays, SciMLBase

# `construct_linear_solver` routes to the native `\` fallback only for scalars, Diagonal,
# an explicit `linsolve = \`, and static arrays (possibly wrapped). A `Symmetric`-wrapped
# dense matrix with the default `linsolve = nothing` must go through LinearSolve so the
# factorization is cached and `reuse_A_if_factorization` works.

stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
b = rand(4)
u = rand(4)

A_dense_sym = Symmetric(rand(4, 4) + 5I)
lc = NonlinearSolveBase.construct_linear_solver(
    nothing, nothing, A_dense_sym, b, u, nothing; stats
)
@test lc isa NonlinearSolveBase.LinearSolveJLCache
res = lc(; A = A_dense_sym, b, linu = u)
@test res.u ≈ A_dense_sym \ b

# Static arrays (bare and wrapped) keep the native fallback
A_smat = SA[5.0 1.0; 1.0 5.0]
bs = SA[1.0, 2.0]
us = SA[0.0, 0.0]
lc_smat = NonlinearSolveBase.construct_linear_solver(
    nothing, nothing, A_smat, bs, us, nothing; stats
)
@test lc_smat isa NonlinearSolveBase.NativeJLLinearSolveCache

lc_smat_sym = NonlinearSolveBase.construct_linear_solver(
    nothing, nothing, Symmetric(A_smat), bs, us, nothing; stats
)
@test lc_smat_sym isa NonlinearSolveBase.NativeJLLinearSolveCache

# Plain dense matrices go to LinearSolve (unchanged behavior)
lc_dense = NonlinearSolveBase.construct_linear_solver(
    nothing, nothing, rand(4, 4) + 5I, b, u, nothing; stats
)
@test lc_dense isa NonlinearSolveBase.LinearSolveJLCache

# `update_A!` must dispatch on the *resolved* algorithm (`lincache.alg`), not the
# user-passed `linsolve` object (which has no `alg` field). With the old dispatch every
# call re-set `A` — factorization algorithms refactorized on every solve even when the
# caller requested reuse via `reuse_A_if_factorization`, and `nfactors` never moved.
stats_reuse = SciMLBase.NLStats(0, 0, 0, 0, 0)
A_fact = rand(4, 4) + 5I
lc_fact = NonlinearSolveBase.construct_linear_solver(
    nothing, LUFactorization(), copy(A_fact), copy(b), copy(u), nothing;
    stats = stats_reuse
)
@test lc_fact isa NonlinearSolveBase.LinearSolveJLCache
res = lc_fact(; A = A_fact, b, linu = u, reuse_A_if_factorization = false)
@test res.u ≈ A_fact \ b
@test stats_reuse.nfactors == 1
# reuse: the factorization must not be redone and nfactors must not move
res = lc_fact(; A = A_fact, b, linu = u, reuse_A_if_factorization = true)
@test res.u ≈ A_fact \ b
@test stats_reuse.nfactors == 1
# fresh A with reuse off refactorizes exactly once more
res = lc_fact(; A = 2 .* A_fact, b, linu = u, reuse_A_if_factorization = false)
@test res.u ≈ (2 .* A_fact) \ b
@test stats_reuse.nfactors == 2
