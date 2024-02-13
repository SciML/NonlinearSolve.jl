function MultiStepNonlinearSolver(; concrete_jac = nothing, linsolve = nothing,
        scheme = MSS.PotraPtak3, precs = DEFAULT_PRECS, autodiff = nothing,
        vjp_autodiff = nothing)
    scheme_concrete = apply_patch(scheme, (; autodiff, vjp_autodiff))
    descent = GenericMultiStepDescent(; scheme = scheme_concrete, linsolve, precs)
    return GeneralizedFirstOrderAlgorithm(; concrete_jac, name = MSS.display_name(scheme),
        descent, jacobian_ad = autodiff)
end
