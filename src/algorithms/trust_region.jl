"""
    TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        radius_update_scheme = RadiusUpdateSchemes.Simple, max_trust_radius::Real = 0 // 1,
        initial_trust_radius::Real = 0 // 1, step_threshold::Real = 1 // 10000,
        shrink_threshold::Real = 1 // 4, expand_threshold::Real = 3 // 4,
        shrink_factor::Real = 1 // 4, expand_factor::Real = 2 // 1,
        max_shrink_times::Int = 32,
        vjp_autodiff = nothing, autodiff = nothing, jvp_autodiff = nothing)

An advanced TrustRegion implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `radius_update_scheme`: the scheme used to update the trust region radius. Defaults to
    `RadiusUpdateSchemes.Simple`. See [`RadiusUpdateSchemes`](@ref) for more details. For a
    review on trust region radius update schemes, see [yuan2015recent](@citet).

For the remaining arguments, see [`NonlinearSolve.GenericTrustRegionScheme`](@ref)
documentation.
"""
function TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        radius_update_scheme = RadiusUpdateSchemes.Simple, max_trust_radius::Real = 0 // 1,
        initial_trust_radius::Real = 0 // 1, step_threshold::Real = 1 // 10000,
        shrink_threshold::Real = 1 // 4, expand_threshold::Real = 3 // 4,
        shrink_factor::Real = 1 // 4, expand_factor::Real = 2 // 1,
        max_shrink_times::Int = 32,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing)
    descent = Dogleg(; linsolve, precs)
    trustregion = GenericTrustRegionScheme(;
        method = radius_update_scheme, step_threshold, shrink_threshold, expand_threshold,
        shrink_factor, expand_factor, initial_trust_radius, max_trust_radius)
    return GeneralizedFirstOrderAlgorithm{concrete_jac, :TrustRegion}(;
        trustregion, descent, autodiff, vjp_autodiff, jvp_autodiff, max_shrink_times)
end
