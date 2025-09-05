module NonlinearSolveBaseMooncakeExt

using NonlinearSolveBase, Mooncake
using SciMLBase: SciMLBase
import Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
                 @from_rrule, @zero_adjoint, @mooncake_overlay, MinimalCtx,
                 NoPullback

@from_rrule(MinimalCtx,
    Tuple{
        typeof(NonlinearSolveBase.solve_up),
        SciMLBase.AbstractDEProblem,
        Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
        Any,
        Any,
        Any
    },
    true,)

# Dispatch for auto-alg
@from_rrule(MinimalCtx,
    Tuple{
        typeof(NonlinearSolveBase.solve_up),
        SciMLBase.AbstractDEProblem,
        Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
        Any,
        Any
    },
    true,)

end
