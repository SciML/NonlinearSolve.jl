module BracketingNonlinearSolveChainRulesCoreExt

using CommonSolve: CommonSolve
using ForwardDiff
using SciMLBase

using BracketingNonlinearSolve: bracketingnonlinear_solve_up, is_extension_loaded

function ChainRulesCore.rrule(
        ::typeof(bracketingnonlinear_solve_up),
        prob::IntervalNonlinearProblem,
        sensealg, p, alg, args...; kwargs...
)
    # DiffEqBase is needed for problem/function constructor adjoint
    out = solve(prob)
    u = out.u
    f = SciMLBase.unwrapped_f(prob.f)
    function ∇bracketingnonlinear_solve_up(Δ)
        # Δ = dg/du
        λ = only(ForwardDiff.derivative(u -> f(u, p), only(u)) \ Δ.u)
        dgdp = -λ * ForwardDiff.derivative(p -> f(u, p), only(p))
        return (NoTangent(), NoTangent(), NoTangent(),
            dgdp, NoTangent(),
            ntuple(_ -> NoTangent(), length(args))...)
    end
    return out, ∇bracketingnonlinear_solve_up
end

end