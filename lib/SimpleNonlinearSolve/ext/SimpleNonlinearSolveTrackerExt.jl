module SimpleNonlinearSolveTrackerExt

using NonlinearSolveBase: ImmutableNonlinearProblem, _solve_adjoint
using SciMLBase: TrackerOriginator, NonlinearLeastSquaresProblem, remake

using ArrayInterface: ArrayInterface
using Tracker: Tracker, TrackedArray, TrackedReal

using SimpleNonlinearSolve: SimpleNonlinearSolve

for pType in (ImmutableNonlinearProblem, NonlinearLeastSquaresProblem)
    aTypes = (TrackedArray, AbstractArray{<:TrackedReal}, Any)
    for (uT, pT) in collect(Iterators.product(aTypes, aTypes))[1:(end - 1)]
        @eval function SimpleNonlinearSolve.simplenonlinearsolve_solve_up(
                prob::$(pType), sensealg, u0::$(uT), u0_changed,
                p::$(pT), p_changed, alg, args...; kwargs...
            )
            return Tracker.track(
                SimpleNonlinearSolve.simplenonlinearsolve_solve_up,
                prob, sensealg, ArrayInterface.aos_to_soa(u0), true,
                ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...
            )
        end
    end

    @eval Tracker.@grad function SimpleNonlinearSolve.simplenonlinearsolve_solve_up(
            tprob::$(pType), sensealg, tu0, u0_changed,
            tp, p_changed, alg, args...; kwargs...
        )
        u0, p = Tracker.data(tu0), Tracker.data(tp)
        prob = remake(tprob; u0, p)
        out,
            ∇internal = _solve_adjoint(
            prob, sensealg, u0, p, TrackerOriginator(), alg, args...; kwargs...
        )

        function ∇simplenonlinearsolve_solve_up(Δ)
            ∂prob, ∂sensealg, ∂u0, ∂p, _, ∂args... = ∇internal(Tracker.data(Δ))
            return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
        end

        return out, ∇simplenonlinearsolve_solve_up
    end
end

end
