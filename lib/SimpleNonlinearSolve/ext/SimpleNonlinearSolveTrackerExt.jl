module SimpleNonlinearSolveTrackerExt

using DiffEqBase: DiffEqBase
using SciMLBase: TrackerOriginator, NonlinearProblem, NonlinearLeastSquaresProblem, remake
using SimpleNonlinearSolve: SimpleNonlinearSolve
using Tracker: Tracker, TrackedArray

for pType in (NonlinearProblem, NonlinearLeastSquaresProblem)
    @eval begin
        function SimpleNonlinearSolve.__internal_solve_up(
                prob::$(pType), sensealg, u0::TrackedArray,
                u0_changed, p, p_changed, alg, args...; kwargs...)
            return Tracker.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg,
                u0, u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        function SimpleNonlinearSolve.__internal_solve_up(
                prob::$(pType), sensealg, u0::TrackedArray, u0_changed,
                p::TrackedArray, p_changed, alg, args...; kwargs...)
            return Tracker.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg,
                u0, u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        function SimpleNonlinearSolve.__internal_solve_up(
                prob::$(pType), sensealg, u0, u0_changed,
                p::TrackedArray, p_changed, alg, args...; kwargs...)
            return Tracker.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg,
                u0, u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        Tracker.@grad function SimpleNonlinearSolve.__internal_solve_up(
                _prob::$(pType), sensealg, u0_, u0_changed,
                p_, p_changed, alg, args...; kwargs...)
            u0, p = Tracker.data(u0_), Tracker.data(p_)
            prob = remake(_prob; u0, p)
            out, ∇internal = DiffEqBase._solve_adjoint(
                prob, sensealg, u0, p, TrackerOriginator(), alg, args...; kwargs...)

            function ∇__internal_solve_up(Δ)
                ∂prob, ∂sensealg, ∂u0, ∂p, ∂originator, ∂args... = ∇internal(Δ)
                return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
            end

            return out, ∇__internal_solve_up
        end
    end
end

end
