module Utils

using ArrayInterface: ArrayInterface
using FastClosures: @closure
using LinearAlgebra: Symmetric, norm, dot
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
using SciMLOperators: AbstractSciMLOperator
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NonlinearFunction
using StaticArraysCore: StaticArray, SArray

using ..NonlinearSolveBase: L2_NORM, Linf_NORM

is_extension_loaded(::Val) = false

fast_scalar_indexing(xs...) = all(ArrayInterface.fast_scalar_indexing, xs)

function nonallocating_isapprox(x::Number, y::Number; atol = false,
        rtol = atol > 0 ? false : sqrt(eps(promote_type(typeof(x), typeof(y)))))
    return isapprox(x, y; atol, rtol)
end
function nonallocating_isapprox(x::AbstractArray, y::AbstractArray; atol = false,
        rtol = atol > 0 ? false : sqrt(eps(eltype(x))))
    length(x) == length(y) || return false
    d = nonallocating_maximum(-, x, y)
    return d ≤ max(atol, rtol * max(maximum(abs, x), maximum(abs, y)))
end

function nonallocating_maximum(f::F, x, y) where {F}
    if fast_scalar_indexing(x, y)
        return maximum(@closure((xᵢyᵢ)->begin
                xᵢ, yᵢ = xᵢyᵢ
                return abs(f(xᵢ, yᵢ))
            end), zip(x, y))
    else
        return mapreduce(@closure((xᵢ, yᵢ)->abs(f(xᵢ, yᵢ))), max, x, y)
    end
end

function abs2_and_sum(x, y)
    return reduce(Base.add_sum, x, init = zero(real(value(eltype(x))))) +
           reduce(Base.add_sum, y, init = zero(real(value(eltype(y)))))
end

children(x::AbstractVectorOfArray) = x.u
children(x::ArrayPartition) = x.x

value(::Type{T}) where {T} = T
value(x) = x

zero_init(x) = zero(real(value(eltype(x))))
one_init(x) = one(real(value(eltype(x))))

standardize_norm(::typeof(Base.Fix1(maximum, abs))) = Linf_NORM
standardize_norm(::typeof(Base.Fix2(norm, Inf))) = Linf_NORM
standardize_norm(::typeof(Base.Fix2(norm, 2))) = L2_NORM
standardize_norm(::typeof(norm)) = L2_NORM
standardize_norm(f::F) where {F} = f

norm_op(norm::N, op::OP, x, y) where {N, OP} = norm(op.(x, y))
function norm_op(::typeof(L2_NORM), op::OP, x, y) where {OP}
    if fast_scalar_indexing(x, y)
        return sqrt(sum(@closure((xᵢyᵢ)->begin
                xᵢ, yᵢ = xᵢyᵢ
                return op(xᵢ, yᵢ)^2
            end), zip(x, y)))
    else
        return sqrt(mapreduce(@closure((xᵢ, yᵢ)->op(xᵢ, yᵢ)^2), +, x, y))
    end
end
function norm_op(::typeof(Linf_NORM), op::OP, x, y) where {OP}
    return nonallocating_maximum(abs ∘ op, x, y)
end

apply_norm(f::F, x) where {F} = standardize_norm(f)(x)
apply_norm(f::F, x, y) where {F} = norm_op(standardize_norm(f), +, x, y)

convert_real(::Type{T}, ::Nothing) where {T} = nothing
convert_real(::Type{T}, x) where {T} = real(T(x))

restructure(::Number, x::Number) = x
restructure(y, x) = ArrayInterface.restructure(y, x)

function safe_similar(x, args...; kwargs...)
    y = similar(x, args...; kwargs...)
    return init_bigfloat_array!!(y)
end

init_bigfloat_array!!(x) = x

function init_bigfloat_array!!(x::AbstractArray{<:BigFloat})
    ArrayInterface.can_setindex(x) && fill!(x, BigFloat(0))
    return x
end

safe_reshape(x::Number, args...) = x
safe_reshape(x, args...) = reshape(x, args...)

@generated function safe_getproperty(s::S, ::Val{X}) where {S, X}
    hasfield(S, X) && return :(getproperty(s, $(Meta.quot(X))))
    return :(missing)
end

@generated function safe_vec(v)
    hasmethod(vec, Tuple{typeof(v)}) || return :(vec(v))
    return :(v)
end
safe_vec(v::Number) = v
safe_vec(v::AbstractVector) = v

safe_dot(x, y) = dot(safe_vec(x), safe_vec(y))

unwrap_val(x) = x
unwrap_val(::Val{x}) where {x} = unwrap_val(x)

is_default_value(::Any, ::Symbol, ::Nothing) = true
is_default_value(::Any, ::Symbol, ::Missing) = true
is_default_value(::Any, ::Symbol, ::Any) = false

maybe_symmetric(x) = Symmetric(x)
maybe_symmetric(x::Number) = x
## LinearSolve with `nothing` doesn't dispatch correctly here
maybe_symmetric(x::StaticArray) = x # XXX: Can we remove this?
maybe_symmetric(x::AbstractSciMLOperator) = x

# Define special concatenation for certain Array combinations
faster_vcat(x, y) = vcat(x, y)

maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
function maybe_unaliased(x::AbstractArray, alias::Bool)
    (alias || !ArrayInterface.can_setindex(typeof(x))) && return x
    return copy(x)
end
maybe_unaliased(x::AbstractSciMLOperator, ::Bool) = x

can_setindex(x) = ArrayInterface.can_setindex(x)
can_setindex(::Number) = false

evaluate_f!!(prob::AbstractNonlinearProblem, fu, u, p) = evaluate_f!!(prob.f, fu, u, p)
function evaluate_f!!(f::NonlinearFunction, fu, u, p)
    if SciMLBase.isinplace(f)
        f(fu, u, p)
        return fu
    end
    return f(u, p)
end

end
