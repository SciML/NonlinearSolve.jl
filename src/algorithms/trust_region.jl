function TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        radius_update_scheme = RadiusUpdateSchemes.Simple, max_trust_radius::Real = 0 // 1,
        initial_trust_radius::Real = 0 // 1, step_threshold::Real = 1 // 10000,
        shrink_threshold::Real = 1 // 4, expand_threshold::Real = 3 // 4,
        shrink_factor::Real = 1 // 4, expand_factor::Real = 2 // 1,
        max_shrink_times::Int = 32, vjp_autodiff = nothing, autodiff = nothing)
    # TODO: max_shrink_times is not used
    descent = Dogleg(; linsolve, precs)
    trustregion = GenericTrustRegionScheme(; method = radius_update_scheme, step_threshold,
        shrink_threshold, expand_threshold, shrink_factor, expand_factor,
        reverse_ad = vjp_autodiff)
    return GeneralizedFirstOrderRootFindingAlgorithm(; concrete_jac, name = :TrustRegion,
        trustregion, descent, jacobian_ad = autodiff)
end
