module BracketingNonlinearSolveChainRulesCoreExt

using BracketingNonlinearSolve: bracketingnonlinear_solve_up, CommonSolve, SciMLBase
using CommonSolve: solve
using SciMLBase: IntervalNonlinearProblem
using ForwardDiff
using ChainRulesCore: ChainRulesCore, AbstractThunk, NoTangent, Tangent, unthunk

function ChainRulesCore.rrule(
        ::typeof(bracketingnonlinear_solve_up),
        prob::IntervalNonlinearProblem,
        sensealg, p, alg, args...; kwargs...
)
    out = solve(prob, alg)
    u = out.u
    f = SciMLBase.unwrapped_f(prob.f)
    function ∇bracketingnonlinear_solve_up(Δ)
        Δ = Δ isa AbstractThunk ? unthunk(Δ) : Δ
        # Δ = dg/du
        Δ isa Tangent ? delu = Δ.u : delu = Δ
        λ = only(ForwardDiff.derivative(u -> f(u, p), only(u)) \ delu)
        if p isa Number
            dgdp = -λ * ForwardDiff.derivative(p -> f(u, p), p)
        else
            dgdp = -λ * ForwardDiff.gradient(p -> f(u, p), p)
        end
        return (NoTangent(), NoTangent(), NoTangent(),
            dgdp, NoTangent(),
            ntuple(_ -> NoTangent(), length(args))...)
    end
    return out, ∇bracketingnonlinear_solve_up
end

end