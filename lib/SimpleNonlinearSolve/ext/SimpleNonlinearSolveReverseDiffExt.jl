module SimpleNonlinearSolveReverseDiffExt

using ArrayInterface: ArrayInterface
using DiffEqBase: DiffEqBase
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal
using SciMLBase: ReverseDiffOriginator, NonlinearProblem, NonlinearLeastSquaresProblem
using SimpleNonlinearSolve: SimpleNonlinearSolve
import SimpleNonlinearSolve: __internal_solve_up

for pType in (NonlinearProblem, NonlinearLeastSquaresProblem)
    @eval begin
        function __internal_solve_up(prob::$(pType), sensealg, u0::TrackedArray, u0_changed,
                p::TrackedArray, p_changed, alg, args...; kwargs...)
            return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
                u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        function __internal_solve_up(prob::$(pType), sensealg, u0, u0_changed,
                p::TrackedArray, p_changed, alg, args...; kwargs...)
            return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
                u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        function __internal_solve_up(prob::$(pType), sensealg, u0::TrackedArray,
                u0_changed, p, p_changed, alg, args...; kwargs...)
            return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
                u0_changed, p, p_changed, alg, args...; kwargs...)
        end

        function __internal_solve_up(
                prob::$(pType), sensealg, u0::AbstractArray{<:TrackedReal}, u0_changed,
                p::AbstractArray{<:TrackedReal}, p_changed, alg, args...; kwargs...)
            return __internal_solve_up(prob, sensealg, ArrayInterface.aos_to_soa(u0), true,
                ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
        end

        function __internal_solve_up(prob::$(pType), sensealg, u0, u0_changed,
                p::AbstractArray{<:TrackedReal}, p_changed, alg, args...; kwargs...)
            return __internal_solve_up(
                prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p),
                true, alg, args...; kwargs...)
        end

        function __internal_solve_up(
                prob::$(pType), sensealg, u0::AbstractArray{<:TrackedReal},
                u0_changed, p, p_changed, alg, args...; kwargs...)
            return __internal_solve_up(
                prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p),
                true, alg, args...; kwargs...)
        end

        ReverseDiff.@grad function __internal_solve_up(
                prob::$(pType), sensealg, u0, u0_changed,
                p, p_changed, alg, args...; kwargs...)
            out, ∇internal = DiffEqBase._solve_adjoint(
                prob, sensealg, ReverseDiff.value(u0), ReverseDiff.value(p),
                ReverseDiffOriginator(), alg, args...; kwargs...)
            function ∇__internal_solve_up(_args...)
                ∂prob, ∂sensealg, ∂u0, ∂p, ∂originator, ∂args... = ∇internal(_args...)
                return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
            end
            return Array(out), ∇__internal_solve_up
        end
    end
end

end
