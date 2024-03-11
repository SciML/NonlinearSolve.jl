module SimpleNonlinearSolveReverseDiffExt

using ArrayInterface, DiffEqBase, ReverseDiff, SciMLBase, SimpleNonlinearSolve
import ReverseDiff: TrackedArray, TrackedReal
import SimpleNonlinearSolve: __internal_solve_up

function __internal_solve_up(
        prob::NonlinearProblem, sensealg, u0::TrackedArray, u0_changed,
        p::TrackedArray, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function __internal_solve_up(
        prob::NonlinearProblem, sensealg, u0, u0_changed,
        p::TrackedArray, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function __internal_solve_up(
        prob::NonlinearProblem, sensealg, u0::TrackedArray, u0_changed,
        p, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function __internal_solve_up(prob::NonlinearProblem, sensealg,
        u0::AbstractArray{<:TrackedReal}, u0_changed, p::AbstractArray{<:TrackedReal},
        p_changed, alg, args...; kwargs...)
    return __internal_solve_up(
        prob, sensealg, ArrayInterface.aos_to_soa(u0), true,
        ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
end

function __internal_solve_up(prob::NonlinearProblem, sensealg, u0, u0_changed,
        p::AbstractArray{<:TrackedReal}, p_changed, alg, args...; kwargs...)
    return __internal_solve_up(
        prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
end

function __internal_solve_up(prob::NonlinearProblem, sensealg,
        u0::AbstractArray{<:TrackedReal}, u0_changed, p, p_changed, alg, args...; kwargs...)
    return __internal_solve_up(
        prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
end

ReverseDiff.@grad function __internal_solve_up(
        prob::NonlinearProblem, sensealg, u0, u0_changed, p, p_changed, alg, args...; kwargs...)
    out, ∇internal = DiffEqBase._solve_adjoint(
        prob, sensealg, ReverseDiff.value(u0), ReverseDiff.value(p),
        SciMLBase.ReverseDiffOriginator(), alg, args...; kwargs...)
    function ∇__internal_solve_up(_args...)
        ∂prob, ∂sensealg, ∂u0, ∂p, ∂originator, ∂args... = ∇internal(_args...)
        return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
    end
    return Array(out), ∇__internal_solve_up
end

end
