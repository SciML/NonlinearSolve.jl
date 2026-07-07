using NonlinearSolveFirstOrder

using NonlinearSolveFirstOrder, SciMLBase

f(u, p) = u .* u .- p
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)
cache = init(prob, TrustRegion())

# Get the trust region cache
tr_cache = cache.trustregion_cache
@test tr_cache.trust_region == tr_cache.initial_trust_radius

# Solve problem to modify the trust region
sol = solve!(cache)
@test SciMLBase.successful_retcode(sol)
@test tr_cache.trust_region != tr_cache.initial_trust_radius

# Reinitialize and check the trust region was reset
reinit!(cache, [1.0, 1.0]; p = 2.0)
@test tr_cache.trust_region == tr_cache.initial_trust_radius

original_initial_trust_radius = tr_cache.initial_trust_radius
reinit!(cache, [100.0, 1.0]; p = 2.0)
@test tr_cache.initial_trust_radius != original_initial_trust_radius
@test tr_cache.initial_trust_radius == tr_cache.max_trust_radius / 11
@test tr_cache.trust_region == tr_cache.initial_trust_radius
