# Ignores NaN
function __findmin(f, x)
    return findmin(x) do xᵢ
        fx = f(xᵢ)
        return isnan(fx) ? Inf : fx
    end
end

@inline value(x) = x
@inline value(x::Dual) = ForwardDiff.value(x)
@inline value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

_mutable_zero(x) = zero(x)
_mutable_zero(x::SArray) = MArray(x)

_mutable(x) = x
_mutable(x::SArray) = MArray(x)

# __maybe_mutable(x, ::AbstractFiniteDifferencesMode) = _mutable(x)
# The shadow allocated for Enzyme needs to be mutable
__maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
__maybe_mutable(x, _) = x

# Helper function to get value of `f(u, p)`
function evaluate_f(f::F, u, p, ::Val{iip}; fu = nothing) where {F, iip}
    if iip
        f(fu, u, p)
        return fu
    else
        return f(u, p)
    end
end

function evaluate_f(cache::AbstractNonlinearSolveCache, u, p,
        fu_sym::Val{FUSYM} = Val(nothing)) where {FUSYM}
    cache.stats.nf += 1
    if FUSYM === nothing
        # if isinplace(cache)
        #     cache.prob.f(get_fu(cache), u, p)
        # else
        #     set_fu!(cache, cache.prob.f(u, p))
        # end
    else
        if isinplace(cache)
            cache.prob.f(__getproperty(cache, fu_sym), u, p)
        else
            setproperty!(cache, FUSYM, cache.prob.f(u, p))
        end
    end
    return nothing
end

function __init_low_rank_jacobian(u::StaticArray{S1, T1}, fu::StaticArray{S2, T2},
        ::Val{threshold}) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
function __init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = similar(u, threshold, length(u))
    U = similar(u, length(fu), threshold)
    return U, Vᵀ
end

# Non-square matrix
@inline __needs_square_A(_, ::Number) = true
@inline __needs_square_A(alg, _) = LinearSolve.needs_square_A(alg.linsolve)

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

# Diagonal of type `u`
__init_diagonal(u::Number, v) = oftype(u, v)
function __init_diagonal(u::SArray, v)
    u_ = vec(u)
    return Diagonal(ones(typeof(u_)) * v)
end
function __init_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end

# Reduce sum
function __sum_JᵀJ!!(y, J)
    if setindex_trait(y) === CanSetindex()
        sum!(abs2, y, J')
        return y
    else
        return sum(abs2, J'; dims = 1)
    end
end

# Alpha for Initial Jacobian Guess
# The values are somewhat different from SciPy, these were tuned to the 23 test problems
@inline function __initial_inv_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), inv(α))
end
@inline function __initial_inv_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(norm_fu ≥ 1e-5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_inv_alpha(inv_α, α::Number, u, fu, norm::F) where {F} = inv_α
@inline function __initial_inv_alpha(inv_α, α::Nothing, u, fu, norm::F) where {F}
    return __initial_inv_alpha(α, u, fu, norm)
end

@inline function __initial_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
@inline function __initial_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(1e-5 ≤ norm_fu ≤ 1e5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_alpha(α_initial, α::Number, u, fu, norm::F) where {F} = α_initial
@inline function __initial_alpha(α_initial, α::Nothing, u, fu, norm::F) where {F}
    return __initial_alpha(α, u, fu, norm)
end

# Diagonal

@inline __is_complex(::Type{ComplexF64}) = true
@inline __is_complex(::Type{ComplexF32}) = true
@inline __is_complex(::Type{Complex}) = true
@inline __is_complex(::Type{T}) where {T} = false

@inline __reshape(x::Number, args...) = x
@inline __reshape(x::AbstractArray, args...) = reshape(x, args...)
