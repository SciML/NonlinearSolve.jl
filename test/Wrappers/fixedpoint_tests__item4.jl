using NonlinearSolve

import NLsolve, SIAMFANLEquations

# N large enough that a dense N×N Float64 matrix would dominate the
# measured allocation, but small enough to solve quickly.
N = 3_200
# Simple fixed point: f(x) = 0 at x_i = 0.739... (cos(x) = x).
F!(F, x, p) = (F .= cos.(x) .- x)
x0 = fill(0.5, N)
prob = NonlinearProblem(NonlinearFunction(F!), x0)

# Cap iterations so per-iteration allocations don't dwarf the one-shot
# Jacobian allocation we're guarding against.
maxit = 10

# Warm up compilation
solve(prob, NLsolveJL(; method = :anderson, m = 5); maxiters = maxit)
solve(prob, SIAMFANLEquationsJL(; method = :anderson, m = 5); maxiters = maxit)

# An N×N Float64 matrix is 8·N² bytes ≈ 82 MB at N=3200.
dense_jac_bytes = 8 * N * N

allocs_nlsolve = @allocated solve(
    prob, NLsolveJL(; method = :anderson, m = 5); maxiters = maxit
)
@test allocs_nlsolve < dense_jac_bytes ÷ 4

allocs_siam = @allocated solve(
    prob, SIAMFANLEquationsJL(; method = :anderson, m = 5); maxiters = maxit
)
@test allocs_siam < dense_jac_bytes ÷ 4
