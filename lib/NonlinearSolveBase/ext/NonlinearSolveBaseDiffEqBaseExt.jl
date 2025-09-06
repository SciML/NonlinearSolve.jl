module NonlinearSolveBaseDiffEqBaseExt

using DiffEqBase: DiffEqBase
using SciMLBase: SciMLBase, remake

using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem

function DiffEqBase.get_concrete_problem(
        prob::ImmutableNonlinearProblem, isadapt; kwargs...)
    u0 = SciMLBase.get_concrete_u0(prob, isadapt, nothing, kwargs)
    u0 = SciMLBase.promote_u0(u0, prob.p, nothing)
    p = SciMLBase.get_concrete_p(prob, kwargs)
    return remake(prob; u0 = u0, p = p)
end

end
