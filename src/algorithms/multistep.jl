function MultiStepNonlinearSolver(; concrete_jac = nothing, linsolve = nothing,
        scheme = MSS.PotraPtak3, precs = DEFAULT_PRECS, autodiff = nothing)
    descent = GenericMultiStepDescent(; scheme, linsolve, precs)
    # TODO: Use the scheme as the name
    return GeneralizedFirstOrderAlgorithm(; concrete_jac, name = :MultiStepNonlinearSolver,
        descent, jacobian_ad = autodiff)
end
