module SimpleNonlinearSolveReverseDiffExt

using NonlinearSolveBase: ImmutableNonlinearProblem, _solve_adjoint
using SciMLBase: ReverseDiffOriginator, NonlinearLeastSquaresProblem, remake

using ArrayInterface: ArrayInterface
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal

using SimpleNonlinearSolve: SimpleNonlinearSolve
import SimpleNonlinearSolve: simplenonlinearsolve_solve_up

for pType in (ImmutableNonlinearProblem, NonlinearLeastSquaresProblem)
    aTypes = (TrackedArray, AbstractArray{<:TrackedReal}, Any)
    for (uT, pT) in collect(Iterators.product(aTypes, aTypes))[1:(end - 1)]
        @eval function simplenonlinearsolve_solve_up(
                prob::$(pType), sensealg, u0::$(uT), u0_changed,
                p::$(pT), p_changed, alg, args...; kwargs...)
            return ReverseDiff.track(SimpleNonlinearSolve.simplenonlinearsolve_solve_up,
                prob, sensealg, ArrayInterface.aos_to_soa(u0), true,
                ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
        end
    end

    @eval ReverseDiff.@grad function simplenonlinearsolve_solve_up(
            tprob::$(pType), sensealg, tu0, u0_changed,
            tp, p_changed, alg, args...; kwargs...)
        u0, p = ReverseDiff.value(tu0), ReverseDiff.value(tp)
        prob = remake(tprob; u0, p)
        out,
        ∇internal = _solve_adjoint(
            prob, sensealg, u0, p, ReverseDiffOriginator(), alg, args...; kwargs...)

        function ∇simplenonlinearsolve_solve_up(Δ...)
            ∂prob, ∂sensealg, ∂u0, ∂p, _, ∂args... = ∇internal(Δ...)
            return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
        end

        return Array(out), ∇simplenonlinearsolve_solve_up
    end
end

end
