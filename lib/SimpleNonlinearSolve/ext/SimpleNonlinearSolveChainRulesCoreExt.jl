module SimpleNonlinearSolveChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent

using NonlinearSolveBase: _solve_adjoint
using SciMLBase: ImmutableNonlinearProblem, ChainRulesOriginator,
    NonlinearLeastSquaresProblem

using SimpleNonlinearSolve: SimpleNonlinearSolve, simplenonlinearsolve_solve_up

function ChainRulesCore.rrule(
        ::typeof(simplenonlinearsolve_solve_up),
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        sensealg, u0, u0_changed, p, p_changed, alg, args...; kwargs...
    )
    out,
        ∇internal = _solve_adjoint(
        prob, sensealg, u0, p, ChainRulesOriginator(), alg, args...; kwargs...
    )
    function ∇simplenonlinearsolve_solve_up(Δ)
        ∂f, ∂prob, ∂sensealg, ∂u0, ∂p, _, ∂args... = ∇internal(Δ)
        return (
            ∂f, ∂prob, ∂sensealg, ∂u0, NoTangent(), ∂p, NoTangent(), NoTangent(), ∂args...,
        )
    end
    return out, ∇simplenonlinearsolve_solve_up
end

end
