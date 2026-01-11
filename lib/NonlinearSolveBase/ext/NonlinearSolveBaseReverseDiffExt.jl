module NonlinearSolveBaseReverseDiffExt

using NonlinearSolveBase
import SciMLBase: SciMLBase, value
import ReverseDiff
import ArrayInterface

# `ReverseDiff.TrackedArray`
function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0::ReverseDiff.TrackedArray,
        p::ReverseDiff.TrackedArray, args...; kwargs...
    )
    return ReverseDiff.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0, p::ReverseDiff.TrackedArray,
        args...; kwargs...
    )
    return ReverseDiff.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0::ReverseDiff.TrackedArray, p,
        args...; kwargs...
    )
    return ReverseDiff.track(NonlinearSolveBase.solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

# `AbstractArray{<:ReverseDiff.TrackedReal}`
function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        },
        u0::AbstractArray{<:ReverseDiff.TrackedReal},
        p::AbstractArray{<:ReverseDiff.TrackedReal}, args...;
        kwargs...
    )
    return NonlinearSolveBase.solve_up(
        prob, sensealg, ArrayInterface.aos_to_soa(u0),
        ArrayInterface.aos_to_soa(p), args...;
        kwargs...
    )
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0,
        p::AbstractArray{<:ReverseDiff.TrackedReal},
        args...; kwargs...
    )
    return NonlinearSolveBase.solve_up(
        prob, sensealg, u0, ArrayInterface.aos_to_soa(p), args...; kwargs...
    )
end

function NonlinearSolveBase.solve_up(
        prob::SciMLBase.NonlinearProblem,
        sensealg::Union{
            SciMLBase.AbstractOverloadingSensitivityAlgorithm,
            Nothing,
        }, u0::ReverseDiff.TrackedArray,
        p::AbstractArray{<:ReverseDiff.TrackedReal},
        args...; kwargs...
    )
    return NonlinearSolveBase.solve_up(
        prob, sensealg, u0, ArrayInterface.aos_to_soa(p), args...; kwargs...
    )
end

# function NonlinearSolveBase.solve_up(prob::SciMLBase.DEProblem,
#         sensealg::Union{
#             SciMLBase.AbstractOverloadingSensitivityAlgorithm,
#             Nothing},
#         u0::AbstractArray{<:ReverseDiff.TrackedReal}, p,
#         args...; kwargs...)
#     NonlinearSolveBase.solve_up(
#         prob, sensealg, ArrayInterface.aos_to_soa(u0), p, args...; kwargs...)
# end

# function NonlinearSolveBase.solve_up(prob::SciMLBase.DEProblem,
#         sensealg::Union{
#             SciMLBase.AbstractOverloadingSensitivityAlgorithm,
#             Nothing},
#         u0::AbstractArray{<:ReverseDiff.TrackedReal}, p::ReverseDiff.TrackedArray,
#         args...; kwargs...)
#     NonlinearSolveBase.solve_up(
#         prob, sensealg, ArrayInterface.aos_to_soa(u0), p, args...; kwargs...)
# end

# Required because ReverseDiff.@grad function SciMLBase.solve_up is not supported!
import NonlinearSolveBase: solve_up
ReverseDiff.@grad function solve_up(prob, sensealg, u0, p, args...; kwargs...)
    out = NonlinearSolveBase._solve_adjoint(
        prob, sensealg, ReverseDiff.value(u0),
        ReverseDiff.value(p),
        SciMLBase.ReverseDiffOriginator(), args...; kwargs...
    )
    function actual_adjoint(_args...)
        original_adjoint = out[2](_args...)
        if isempty(args) # alg is missing
            tuple(original_adjoint[1:4]..., original_adjoint[6:end]...)
        else
            original_adjoint
        end
    end
    Array(out[1]), actual_adjoint
end

end
