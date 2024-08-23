module NonlinearSolveBaseDiffEqBaseExt

using DiffEqBase: DiffEqBase
using NonlinearSolveBase: ImmutableNonlinearProblem

function DiffEqBase.get_concrete_problem(
        prob::ImmutableNonlinearProblem, isadapt; kwargs...)
    u0 = DiffEqBase.get_concrete_u0(prob, isadapt, nothing, kwargs)
    u0 = DiffEqBase.promote_u0(u0, prob.p, nothing)
    p = DiffEqBase.get_concrete_p(prob, kwargs)
    return SciMLBase.remake(prob; u0, p)
end

end
