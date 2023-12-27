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

# TODO: __concrete_jac
# __concrete_jac(_) = nothing
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
