using NonlinearSolveBase: NonlinearSolveBase, InternalAPI, NewtonDescent, SteepestDescent,
    Utils
using SciMLBase: SciMLBase, NonlinearProblem, NLStats
using LinearAlgebra
using Test

# `similar` on a plain `Array` usually happens to hand back zeroed pages, which hides an
# unwritten buffer; this type makes "unwritten" observable.
struct DirtyVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(v::DirtyVector) = size(v.data)
Base.getindex(v::DirtyVector, i::Int) = v.data[i]
Base.setindex!(v::DirtyVector, x, i::Int) = (v.data[i] = x; v)
function Base.similar(::DirtyVector, ::Type{T}, dims::Dims{1}) where {T <: Number}
    return DirtyVector(fill(T(1.0e30), dims))
end

# A descent buffer is passed to LinearSolve as `linu`, which iterative algorithms read as
# the initial guess *before* the first solve writes it (`WarmStart.Previous`/`Hegedus`).
# Leaving it unwritten makes the linear solve, and hence any integrator driving it, depend
# on whatever the allocator last left in that memory.
@testset "descent caches start with a defined δu" begin
    n = 4
    u = DirtyVector(collect(1.0:n))
    fu = DirtyVector(fill(0.5, n))
    J = Matrix{Float64}(I, n, n)
    prob = NonlinearProblem((du, y, p) -> (du .= y), collect(1.0:n))

    # `pre_inverted` is chosen per algorithm so that no LinearSolve cache is built: the
    # buffer under test is allocated before that branch either way.
    for (alg, pre_inverted) in ((NewtonDescent(), Val(true)), (SteepestDescent(), Val(false)))
        for shared in (Val(1), Val(2))
            cache = InternalAPI.init(
                prob, alg, J, fu, u; stats = NLStats(0, 0, 0, 0, 0), shared, pre_inverted
            )
            for idx in 1:Utils.unwrap_val(shared)
                @test iszero(SciMLBase.get_du(cache, Val(idx)))
            end
        end
    end
end
