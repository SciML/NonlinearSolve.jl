# Defaults
@inline DEFAULT_NORM(args...) = DiffEqBase.NONLINEARSOLVE_DEFAULT_NORM(args...)
@inline DEFAULT_PRECS(W, du, u, p, t, newW, Plprev, Prprev, cachedata) = nothing, nothing
@inline DEFAULT_TOLERANCE(args...) = DiffEqBase._get_tolerance(args...)

# Helper  Functions
@static if VERSION ≤ v"1.10-"
    @inline @generated function __hasfield(::T, ::Val{field}) where {T, field}
        return :($(field ∉ fieldnames(T)))
    end
else
    @inline __hasfield(::T, ::Val{field}) where {T, field} = hasfield(T, field)
end

@generated function __getproperty(s::S, ::Val{X}) where {S, X}
    hasfield(S, X) && return :(s.$X)
    return :(missing)
end

@inline __needs_concrete_A(::Nothing) = false
@inline __needs_concrete_A(::typeof(\)) = true
@inline __needs_concrete_A(linsolve) = needs_concrete_A(linsolve)

@inline __maybe_mutable(x, ::AutoSparse{<:AutoEnzyme}) = __mutable(x)  # TODO: remove?
@inline __maybe_mutable(x, _) = x

@inline @generated function _vec(v)
    hasmethod(vec, Tuple{typeof(v)}) || return :(vec(v))
    return :(v)
end
@inline _vec(v::Number) = v
@inline _vec(v::AbstractVector) = v

@inline _restructure(y, x) = restructure(y, x)
@inline _restructure(y::Number, x::Number) = x

@inline function __init_ones(x)
    w = similar(x)
    recursivefill!(w, true)
    return w
end
@inline __init_ones(x::StaticArray) = ones(typeof(x))

@inline __maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
@inline function __maybe_unaliased(x::AbstractArray, alias::Bool)
    # Spend time coping iff we will mutate the array
    (alias || !__can_setindex(typeof(x))) && return x
    return deepcopy(x)
end
@inline __maybe_unaliased(x::AbstractNonlinearSolveOperator, alias::Bool) = x
@inline __maybe_unaliased(x::AbstractJacobianOperator, alias::Bool) = x

@inline __cond(J::AbstractMatrix) = cond(J)
@inline __cond(J::SVector) = __cond(Diagonal(MVector(J)))
@inline __cond(J::AbstractVector) = __cond(Diagonal(J))
@inline __cond(J::ApplyArray) = __cond(J.f(J.args...))
@inline __cond(J::SparseMatrixCSC) = __cond(Matrix(J))
@inline __cond(J) = -1  # Covers cases where `J` is a Operator, nothing, etc.

@inline __copy(x::AbstractArray) = copy(x)
@inline __copy(x::Number) = x
@inline __copy(x) = x

# LazyArrays for tracing
__zero(x::AbstractArray) = zero(x)
__zero(x) = x
LazyArrays.applied_eltype(::typeof(__zero), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__zero), x) = ndims(x)
LazyArrays.applied_size(::typeof(__zero), x) = size(x)
LazyArrays.applied_axes(::typeof(__zero), x) = axes(x)

# Use Symmetric Matrices if known to be efficient
@inline __maybe_symmetric(x) = Symmetric(x)
@inline __maybe_symmetric(x::Number) = x
## LinearSolve with `nothing` doesn't dispatch correctly here
@inline __maybe_symmetric(x::StaticArray) = x
@inline __maybe_symmetric(x::AbstractSparseMatrix) = x
@inline __maybe_symmetric(x::AbstractSciMLOperator) = x

# SparseAD --> NonSparseAD
@inline __get_nonsparse_ad(backend::AutoSparse) = ADTypes.dense_ad(backend)
@inline __get_nonsparse_ad(ad) = ad

# Simple Checks
@inline __is_present(::Nothing) = false
@inline __is_present(::Missing) = false
@inline __is_present(::Any) = true
@inline __is_present(::NoLineSearch) = false

@inline __is_complex(::Type{ComplexF64}) = true
@inline __is_complex(::Type{ComplexF32}) = true
@inline __is_complex(::Type{Complex}) = true
@inline __is_complex(::Type{T}) where {T} = false

@inline __findmin_caches(f::F, caches) where {F} = __findmin(f ∘ get_fu, caches)
# FIXME: DEFAULT_NORM makes an Array of NaNs not a NaN (atleast according to `isnan`)
@generated function __findmin(f::F, x) where {F}
    # JET shows dynamic dispatch if this is not written as a generated function
    if F === typeof(DEFAULT_NORM)
        return :(return __findmin_impl(Base.Fix1(maximum, abs), x))
    end
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

@inline __can_setindex(x) = can_setindex(x)
@inline __can_setindex(::Number) = false

@inline function __mutable(x)
    __can_setindex(x) && return x
    y = similar(x)
    copyto!(y, x)
    return y
end
@inline __mutable(x::SArray) = MArray(x)

@inline __dot(x, y) = dot(_vec(x), _vec(y))

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

@inline empty_nlstats() = NLStats(0, 0, 0, 0, 0)
function __reinit_internal!(stats::NLStats)
    stats.nf = 0
    stats.nsteps = 0
    stats.nfactors = 0
    stats.njacs = 0
    stats.nsolve = 0
end

function __similar(x, args...; kwargs...)
    y = similar(x, args...; kwargs...)
    return zero(y)
end
