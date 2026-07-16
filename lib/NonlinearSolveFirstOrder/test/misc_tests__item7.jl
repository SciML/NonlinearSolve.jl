using NonlinearSolveFirstOrder, LineSearch

# The corrector seam is honored under the :LineSearch globalization too, and is
# documented-unsupported (must be inert, never called) under :TrustRegion, which
# produces its update and residual together and offers no post-update seam.
function f!(F, u, p)
    F[1] = u[1] - p[1]
    F[2] = u[2] - u[1]^3
    return nothing
end

u0 = [0.0, 0.0]
p = [1.0]
prob = NonlinearProblem(NonlinearFunction(f!), u0, p)

called = Ref(0)
function corrector!(u, u_prev, p)
    called[] += 1
    u[2] = u[1]^3
    return nothing
end

# --- :LineSearch globalization honors the corrector ---
called[] = 0
sol_ls = solve(
    prob, NewtonRaphson(; linesearch = BackTracking(), corrector = corrector!);
    abstol = 1e-12
)
@test sol_ls.retcode == ReturnCode.Success
@test sol_ls.u[2] == sol_ls.u[1]^3          # corrector applied on the returned iterate
@test called[] == sol_ls.stats.nsteps
@test called[] ≥ 1

# --- :TrustRegion ignores the corrector (documented unsupported) ---
# Build a trust-region GeneralizedFirstOrderAlgorithm carrying a corrector by
# reusing the descent/trustregion from a stock TrustRegion() (public fields) and
# the exported keyword constructor. The corrector must never fire, and the solve
# must converge exactly as a corrector-less trust-region run would.
tr = TrustRegion()
tr_with_corrector = GeneralizedFirstOrderAlgorithm(;
    trustregion = tr.trustregion, descent = tr.descent, corrector = corrector!,
    name = :TrustRegionWithCorrector
)

called[] = 0
sol_tr = solve(prob, tr_with_corrector; abstol = 1e-12)
sol_tr_ref = solve(prob, TrustRegion(); abstol = 1e-12)
@test called[] == 0                          # corrector never invoked under :TrustRegion
@test sol_tr.retcode == ReturnCode.Success
@test sol_tr.u ≈ sol_tr_ref.u
