using NonlinearSolveBase
using SciMLBase
using ADTypes
using ArrayInterface
using ForwardDiff
using Test

struct NoScalarVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::NoScalarVector) = size(x.data)
Base.getindex(x::NoScalarVector, i::Int) = x.data[i]
ArrayInterface.fast_scalar_indexing(::Type{<:NoScalarVector}) = false

prob = NonlinearProblem((u, p) -> u .- p, NoScalarVector([1.0, 2.0]), 1.0)

@test NonlinearSolveBase.select_forward_mode_autodiff(prob, nothing) isa AutoForwardDiff
@test NonlinearSolveBase.select_forward_mode_autodiff(prob, AutoForwardDiff()) isa
    AutoForwardDiff

# Neither FiniteDiff nor Zygote is in this test environment, so no backend qualifies for
# a Jacobian or a pullback even though ForwardDiff (and possibly Enzyme) is loaded.
@test Base.find_package("FiniteDiff") === nothing
@test Base.find_package("Zygote") === nothing
@test_throws r"does not support fast scalar indexing" NonlinearSolveBase.select_jacobian_autodiff(
    prob, nothing
)
@test_throws r"does not support fast scalar indexing" NonlinearSolveBase.select_reverse_mode_autodiff(
    prob, nothing
)

vector_prob = NonlinearProblem((u, p) -> u .- p, [1.0, 2.0], 1.0)
@test NonlinearSolveBase.select_jacobian_autodiff(vector_prob, nothing) isa AutoForwardDiff
