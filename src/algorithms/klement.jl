# TODO: Support alpha
function Klement(; max_resets::UInt = 100, linsolve = nothing, alpha = true,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS,
        init_jacobian::Val{IJ} = Val(:identity), autodiff = nothing) where {IJ}
    if IJ === :identity
        initialization = IdentityInitialization(DiagonalStructure())
    elseif IJ === :true_jacobian
        initialization = TrueJacobianInitialization(FullStructure())
    elseif IJ === :true_jacobian_diagonal
        initialization = TrueJacobianInitialization(DiagonalStructure())
    else
        throw(ArgumentError("`init_jacobian` must be one of `:identity`, `:true_jacobian`, \
                             or `:true_jacobian_diagonal`"))
    end
    return ApproximateJacobianSolveAlgorithm{false}(:Klement, autodiff, initialization,
        NoJacobianDamping(), linesearch, update_rule, reinit_rule, linsolve, precs,
        max_resets)
    #     update_rule::UR
    # reinit_rule
    # linsolve
    # precs
    # max_resets::UInt
end
