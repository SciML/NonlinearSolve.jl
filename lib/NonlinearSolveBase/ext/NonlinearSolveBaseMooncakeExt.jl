module NonlinearSolveBaseMooncakeExt

using NonlinearSolveBase, Mooncake
using SciMLBase: SciMLBase
using Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_chainrules, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback

@from_chainrules MinimalCtx Tuple{
    typeof(NonlinearSolveBase.solve_up),
    SciMLBase.AbstractNonlinearProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
    Any,
} true

# Dispatch for auto-alg
@from_chainrules MinimalCtx Tuple{
    typeof(NonlinearSolveBase.solve_up),
    SciMLBase.AbstractNonlinearProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
} true

end
