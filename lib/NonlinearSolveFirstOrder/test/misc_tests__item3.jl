using NonlinearSolveFirstOrder

using NonlinearSolveFirstOrder, SciMLBase
using ADTypes: AutoEnzyme, AutoForwardDiff
using Enzyme

f(u, p) = u .* u .- p
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)
cache = init(prob, TrustRegion())

# Get the trust region cache
tr_cache = cache.trustregion_cache
@test tr_cache.trust_region == tr_cache.initial_trust_radius

@testset "AutoDespecialize trust region reinitialization" for autodiff in
    (AutoForwardDiff(), AutoEnzyme())
    dynamic_f = NonlinearFunction{false, SciMLBase.AutoDespecialize}(f)
    dynamic_prob = NonlinearProblem(dynamic_f, [1.0, 1.0], 2.0)
    dynamic_cache = init(
        dynamic_prob, TrustRegion(; autodiff, jvp_autodiff = autodiff, vjp_autodiff = autodiff)
    )
    reinit!(dynamic_cache, [1.0, 1.0]; p = 3.0)
    dynamic_sol = solve!(dynamic_cache)
    @test SciMLBase.successful_retcode(dynamic_sol)
    @test dynamic_sol.u ≈ fill(sqrt(3.0), 2)
end

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
