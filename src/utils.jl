@inline __is_complex(::Type{ComplexF64}) = true
@inline __is_complex(::Type{ComplexF32}) = true
@inline __is_complex(::Type{Complex}) = true
@inline __is_complex(::Type{T}) where {T} = false

@inline __findmin_caches(f::F, caches) where {F} = __findmin(f ∘ get_fu, caches)
# FIXME: L2_NORM makes an Array of NaNs not a NaN (atleast according to `isnan`)
@generated function __findmin(f::F, x) where {F}
    # JET shows dynamic dispatch if this is not written as a generated function
    F === typeof(L2_NORM) && return :(return __findmin_impl(Base.Fix1(maximum, abs), x))
    return :(return __findmin_impl(f, x))
end
@inline @views function __findmin_impl(f::F, x) where {F}
    idx = findfirst(Base.Fix2(!==, nothing), x)
    # This is an internal function so we assume that inputs are consistent and there is
    # atleast one non-`nothing` value
    fx_idx = f(x[idx])
    idx == length(x) && return fx_idx, idx
    fmin = @closure xᵢ -> begin
        xᵢ === nothing && return oftype(fx_idx, Inf)
        fx = f(xᵢ)
        return ifelse(isnan(fx), oftype(fx, Inf), fx)
    end
    x_min, x_min_idx = findmin(fmin, x[(idx + 1):length(x)])
    x_min < fx_idx && return x_min, x_min_idx + idx
    return fx_idx, idx
end

"""
    pickchunksize(x) = pickchunksize(length(x))
    pickchunksize(x::Int)

Determine the chunk size for ForwardDiff and PolyesterForwardDiff based on the input length.
"""
@inline pickchunksize(x) = pickchunksize(length(x))
@inline pickchunksize(x::Int) = ForwardDiff.pickchunksize(x)

# Original is often determined on runtime information especially for PolyAlgorithms so it
# is best to never specialize on that
function __build_solution_less_specialize(prob::AbstractNonlinearProblem, alg, u, resid;
        retcode = ReturnCode.Default, original = nothing, left = nothing,
        right = nothing, stats = nothing, trace = nothing, kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    return SciMLBase.NonlinearSolution{
        T, N, typeof(u), typeof(resid), typeof(prob), typeof(alg),
        Any, typeof(left), typeof(stats), typeof(trace)}(
        u, resid, prob, alg, retcode, original, left, right, stats, trace)
end
