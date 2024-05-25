module SimpleNonlinearSolveReverseDiffExt

using ArrayInterface: ArrayInterface
using DiffEqBase: DiffEqBase
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal
using SciMLBase: ReverseDiffOriginator, NonlinearProblem, NonlinearLeastSquaresProblem
using SimpleNonlinearSolve: SimpleNonlinearSolve

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg,
        u0::TrackedArray, u0_changed, p::TrackedArray, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg, u0, u0_changed,
        p::TrackedArray, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg,
        u0::TrackedArray, u0_changed, p, p_changed, alg, args...; kwargs...)
    return ReverseDiff.track(SimpleNonlinearSolve.__internal_solve_up, prob, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
end

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg,
        u0::AbstractArray{<:TrackedReal}, u0_changed, p::AbstractArray{<:TrackedReal},
        p_changed, alg, args...; kwargs...)
    return SimpleNonlinearSolve.__internal_solve_up(
        prob, sensealg, ArrayInterface.aos_to_soa(u0), true,
        ArrayInterface.aos_to_soa(p), true, alg, args...; kwargs...)
end

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg, u0, u0_changed,
        p::AbstractArray{<:TrackedReal}, p_changed, alg, args...; kwargs...)
    return SimpleNonlinearSolve.__internal_solve_up(
        prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p), true, alg, args...;
        kwargs...)
end

function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg,
        u0::AbstractArray{<:TrackedReal}, u0_changed, p, p_changed, alg, args...; kwargs...)
    return SimpleNonlinearSolve.__internal_solve_up(
        prob, sensealg, u0, true, ArrayInterface.aos_to_soa(p), true, alg, args...;
        kwargs...)
end

ReverseDiff.@grad function SimpleNonlinearSolve.__internal_solve_up(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem}, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...)
    out, ∇internal = DiffEqBase._solve_adjoint(
        prob, sensealg, ReverseDiff.value(u0), ReverseDiff.value(p),
        ReverseDiffOriginator(), alg, args...; kwargs...)
    function ∇SimpleNonlinearSolve.__internal_solve_up(_args...)
        ∂prob, ∂sensealg, ∂u0, ∂p, ∂originator, ∂args... = ∇internal(_args...)
        return (∂prob, ∂sensealg, ∂u0, nothing, ∂p, nothing, nothing, ∂args...)
    end
    return Array(out), ∇SimpleNonlinearSolve.__internal_solve_up
end

end
