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
    return :(nothing)
end

@inline __needs_concrete_A(::Nothing) = false
@inline __needs_concrete_A(linsolve) = needs_concrete_A(linsolve)

@inline __maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
@inline __maybe_mutable(x, _) = x

__concrete_jac(::Any) = nothing
# __concrete_jac(::AbstractNewtonAlgorithm{CJ}) where {CJ} = CJ

@inline @generated function _vec(v)
    hasmethod(vec, Tuple{typeof(v)}) || return :(v)
    return :(vec(v))
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
    (alias || !can_setindex(typeof(x))) && return x
    return deepcopy(x)
end

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
@inline __maybe_symmetric(x::SparseArrays.AbstractSparseMatrix) = x
@inline __maybe_symmetric(x::SciMLOperators.AbstractSciMLOperator) = x
