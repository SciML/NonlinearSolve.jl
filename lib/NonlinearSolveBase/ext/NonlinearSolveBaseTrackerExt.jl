module NonlinearSolveBaseTrackerExt

using NonlinearSolveBase
import SciMLBase: SciMLBase, value
import Tracker

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0::Tracker.TrackedArray,
        p::Tracker.TrackedArray, args...; kwargs...
    )
    return Tracker.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0::Tracker.TrackedArray, p, args...;
        kwargs...
    )
    return Tracker.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0, p::Tracker.TrackedArray, args...;
        kwargs...
    )
    return Tracker.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

Tracker.@grad function NonlinearSolveBase.solve_up(
        prob,
        sensealg::Union{
            Nothing,
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
        },
        u0, p, args...;
        kwargs...
    )
    sol,
        pb_f = NonlinearSolveBase._solve_adjoint(
        prob, sensealg, Tracker.data(u0), Tracker.data(p),
        SciMLBase.TrackerOriginator(), args...; kwargs...
    )

    if sol isa AbstractArray
        !hasfield(typeof(sol), :u) && return sol, pb_f # being safe here
        return sol.u, pb_f # AbstractNoTimeSolution isa AbstractArray
    end
    return convert(AbstractArray, sol), pb_f
end

end
