module Utils

using ArrayInterface: ArrayInterface
using ConcreteStructs: @concrete
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, norm, dot, cond, diagind, pinv
using MaybeInplace: @bb
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition, recursivecopy!
using SciMLOperators: AbstractSciMLOperator
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NonlinearFunction
using StaticArraysCore: StaticArray, SArray, SMatrix

using ..NonlinearSolveBase: NonlinearSolveBase, L2_NORM, Linf_NORM

is_extension_loaded(::Val) = false

fast_scalar_indexing(xs...) = all(ArrayInterface.fast_scalar_indexing, xs)

@concrete struct Pinv
    J
end

function Base.convert(::Type{AbstractArray}, A::Pinv)
    hasmethod(pinv, Tuple{typeof(A.J)}) && return pinv(A.J)
    @warn "`pinv` not defined for $(typeof(A.J)). Jacobian will not be inverted when \
           tracing." maxlog=1
    return A.J
end

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
function restructure(
        y::T1, x::T2
) where {T1 <: AbstractSciMLOperator, T2 <: AbstractSciMLOperator}
    @assert size(y)==size(x) "cannot restructure operators. ensure their sizes match."
    return x
end
restructure(y, x) = ArrayInterface.restructure(y, x)

function safe_similar(x, args...; kwargs...)
    y = similar(x, args...; kwargs...)
    return init_similar_array!!(y)
end

init_similar_array!!(x) = x

function init_similar_array!!(x::AbstractArray{<:T}) where {T <: Number}
    ArrayInterface.can_setindex(x) && fill!(x, T(0))
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
is_default_value(::Any, ::Symbol, val::Int) = val == typemax(typeof(val))
is_default_value(::Any, ::Symbol, ::Any) = false
is_default_value(::Any, ::Any, ::Any) = false

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

function evaluate_f!!(prob::AbstractNonlinearProblem, fu, u, p = prob.p)
    return evaluate_f!!(prob.f, fu, u, p)
end
function evaluate_f!!(f::NonlinearFunction, fu, u, p)
    if SciMLBase.isinplace(f)
        f(fu, u, p)
        return fu
    end
    return f(u, p)
end

function evaluate_f(prob::AbstractNonlinearProblem, u)
    if SciMLBase.isinplace(prob)
        fu = prob.f.resid_prototype === nothing ? similar(u) :
             similar(prob.f.resid_prototype)
        prob.f(fu, u, prob.p)
    else
        fu = prob.f(u, prob.p)
    end
    return fu
end

function evaluate_f!(cache, u, p)
    cache.stats.nf += 1
    if SciMLBase.isinplace(cache)
        cache.prob.f(NonlinearSolveBase.get_fu(cache), u, p)
    else
        NonlinearSolveBase.set_fu!(cache, cache.prob.f(u, p))
    end
end

# make_sparse function declaration - implementation provided by SparseArrays extension
# When SparseArrays is not loaded, this function should not be called
function make_sparse end

condition_number(J::AbstractMatrix) = cond(J)
function condition_number(J::AbstractVector)
    if !ArrayInterface.can_setindex(J)
        J′ = similar(J)
        copyto!(J′, J)
        J = J′
    end
    return cond(Diagonal(J))
end
condition_number(::Any) = -1

# compute `pinv` if `inv` won't work
maybe_pinv!!_workspace(A) = nothing, A

maybe_pinv!!(workspace, A::Union{Number, AbstractMatrix}) = pinv(A)
function maybe_pinv!!(workspace, A::Diagonal)
    D = A.diag
    @bb @. D = pinv(D)
    return Diagonal(D)
end
maybe_pinv!!(workspace, A::AbstractVector) = maybe_pinv!!(workspace, Diagonal(A))
function maybe_pinv!!(workspace, A::StridedMatrix)
    LinearAlgebra.checksquare(A)
    if LinearAlgebra.istriu(A)
        issingular = any(iszero, @view(A[diagind(A)]))
        A_ = LinearAlgebra.UpperTriangular(A)
        !issingular && return LinearAlgebra.triu!(parent(inv(A_)))
    elseif LinearAlgebra.istril(A)
        A_ = LinearAlgebra.LowerTriangular(A)
        issingular = any(iszero, @view(A_[diagind(A_)]))
        !issingular && return LinearAlgebra.tril!(parent(inv(A_)))
    else
        F = LinearAlgebra.lu(A; check = false)
        if LinearAlgebra.issuccess(F)
            Ai = LinearAlgebra.inv!(F)
            return convert(typeof(parent(Ai)), Ai)
        end
    end
    return pinv(A)
end

function initial_jacobian_scaling_alpha(α, u, fu, ::Any)
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
function initial_jacobian_scaling_alpha(::Nothing, u, fu, internalnorm::F) where {F}
    fu_norm = internalnorm(fu)
    fu_norm < 1e-5 && return initial_jacobian_scaling_alpha(true, u, fu, internalnorm)
    return (2 * fu_norm) / max(L2_NORM(u), true)
end

make_identity!!(::T, α) where {T <: Number} = T(α)
function make_identity!!(A::AbstractVector{T}, α) where {T}
    if ArrayInterface.can_setindex(A)
        @. A = α
    else
        A = one.(A) .* α
    end
    return A
end
function make_identity!!(::SMatrix{S1, S2, T, L}, α) where {S1, S2, T, L}
    return SMatrix{S1, S2, T, L}(LinearAlgebra.I * α)
end
function make_identity!!(A::AbstractMatrix{T}, α) where {T}
    A = ArrayInterface.can_setindex(A) ? A : similar(A)
    fill!(A, false)
    if ArrayInterface.fast_scalar_indexing(A)
        @simd ivdep for i in axes(A, 1)
            @inbounds A[i, i] = α
        end
    else
        A[diagind(A)] .= α
    end
    return A
end

function reinit_common!(cache, u0, p, alias_u0::Bool)
    if SciMLBase.isinplace(cache)
        recursivecopy!(cache.u, u0)
        cache.prob.f(cache.fu, cache.u, p)
    else
        cache.u = maybe_unaliased(u0, alias_u0)
        NonlinearSolveBase.set_fu!(cache, cache.prob.f(u0, p))
    end
    cache.p = p
end

function clean_sprint_struct(x)
    x isa Symbol && return "$(Meta.quot(x))"
    x isa Number && return string(x)
    (!Base.isstructtype(typeof(x)) || x isa Val) && return string(x)

    modifiers = String[]
    name = nameof(typeof(x))
    for field in fieldnames(typeof(x))
        val = getfield(x, field)
        if field === :name && val isa Symbol && val !== :unknown
            name = val
            continue
        end
        is_default_value(x, field, val) && continue
        push!(modifiers, "$(field) = $(clean_sprint_struct(val))")
    end

    return "$(nameof(typeof(x)))($(join(modifiers, ", ")))"
end

function clean_sprint_struct(x, indent::Int)
    x isa Symbol && return "$(Meta.quot(x))"
    x isa Number && return string(x)
    (!Base.isstructtype(typeof(x)) || x isa Val) && return string(x)

    modifiers = String[]
    name = nameof(typeof(x))
    for field in fieldnames(typeof(x))
        val = getfield(x, field)
        if field === :name && val isa Symbol && val !== :unknown
            name = val
            continue
        end
        is_default_value(x, field, val) && continue
        push!(modifiers, "$(field) = $(clean_sprint_struct(val, indent + 4))")
    end
    spacing = " "^indent * "    "
    spacing_last = " "^indent

    length(modifiers) == 0 && return "$(nameof(typeof(x)))()"
    return "$(name)(\n$(spacing)$(join(modifiers, ",\n$(spacing)"))\n$(spacing_last))"
end

end
