module SciMLJacobianOperators

using ConcreteStructs: @concrete
using ConstructionBase: ConstructionBase
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra
using SciMLBase: SciMLBase, AbstractNonlinearProblem, AbstractNonlinearFunction
using SciMLOperators: AbstractSciMLOperator
using Setfield: @set!

const DI = DifferentiationInterface
const True = Val(true)
const False = Val(false)

abstract type AbstractJacobianOperator{T} <: AbstractSciMLOperator{T} end

abstract type AbstractMode end

struct VJP <: AbstractMode end
struct JVP <: AbstractMode end

flip_mode(::VJP) = JVP()
flip_mode(::JVP) = VJP()

"""
    JacobianOperator{iip, T} <: AbstractJacobianOperator{T} <: AbstractSciMLOperator{T}

A Jacobian Operator Provides both JVP and VJP without materializing either (if possible).

### Constructor

```julia
JacobianOperator(prob::AbstractNonlinearProblem, fu, u; jvp_autodiff = nothing,
    vjp_autodiff = nothing, skip_vjp::Val = Val(false), skip_jvp::Val = Val(false))
```

By default, the `JacobianOperator` will compute `JVP`. Use `Base.adjoint` or
`Base.transpose` to switch to `VJP`.

### Computing the VJP

Computing the VJP is done according to the following rules:

  - If `f` has a `vjp` method, then we use that.
  - If `f` has a `jac` method and no `vjp_autodiff` is provided, then we use `jac * v`.
  - If `vjp_autodiff` is provided we using DifferentiationInterface.jl to compute the VJP.

### Computing the JVP

Computing the JVP is done according to the following rules:

  - If `f` has a `jvp` method, then we use that.
  - If `f` has a `jac` method and no `jvp_autodiff` is provided, then we use `v * jac`.
  - If `jvp_autodiff` is provided we using DifferentiationInterface.jl to compute the JVP.

### Special Case (Number)

For Number inputs, VJP and JVP are not distinct. Hence, if either `vjp` or `jvp` is
provided, then we use that. If neither is provided, then we use `v * jac` if `jac` is
provided. Finally, we use the respective autodiff methods to compute the derivative
using DifferentiationInterface.jl and multiply by `v`.

### Methods Provided

!!! warning

    Currently it is expected that `p` during problem construction is same as `p` during
    operator evaluation. This restriction will be lifted in the future.

  - `(op::JacobianOperator)(v, u, p)`: Computes `∂f(u, p)/∂u * v` or `∂f(u, p)/∂uᵀ * v`.
  - `(op::JacobianOperator)(res, v, u, p)`: Computes `∂f(u, p)/∂u * v` or `∂f(u, p)/∂uᵀ * v`
    and stores the result in `res`.

See also [`VecJacOperator`](@ref) and [`JacVecOperator`](@ref).
"""
@concrete struct JacobianOperator{iip, T <: Real} <: AbstractJacobianOperator{T}
    mode <: AbstractMode

    jvp_op
    vjp_op

    size

    output_cache
    input_cache
end

SciMLBase.isinplace(::JacobianOperator{iip}) where {iip} = iip

function ConstructionBase.constructorof(::Type{<:JacobianOperator{iip, T}}) where {iip, T}
    return JacobianOperator{iip, T}
end

Base.size(J::JacobianOperator) = J.size
Base.size(J::JacobianOperator, d::Integer) = J.size[d]

for op in (:adjoint, :transpose)
    @eval function Base.$(op)(operator::JacobianOperator)
        @set! operator.mode = flip_mode(operator.mode)
        (; output_cache, input_cache) = operator
        @set! operator.output_cache = input_cache
        @set! operator.input_cache = output_cache
        return operator
    end
end

function JacobianOperator(prob::AbstractNonlinearProblem, fu, u; jvp_autodiff = nothing,
        vjp_autodiff = nothing, skip_vjp::Val = False, skip_jvp::Val = False)
    @assert !(skip_vjp === True && skip_jvp === True) "Cannot skip both vjp and jvp \
                                                       construction."
    f = prob.f
    iip = SciMLBase.isinplace(prob)
    T = promote_type(eltype(u), eltype(fu))

    vjp_op = prepare_vjp(skip_vjp, prob, f, u, fu; autodiff = vjp_autodiff)
    jvp_op = prepare_jvp(skip_jvp, prob, f, u, fu; autodiff = jvp_autodiff)

    output_cache = fu isa Number ? T(fu) : similar(fu, T)
    input_cache = u isa Number ? T(u) : similar(u, T)

    return JacobianOperator{iip, T}(
        JVP(), jvp_op, vjp_op, (length(fu), length(u)), output_cache, input_cache)
end

function (op::JacobianOperator)(v, u, p)
    if op.mode isa VJP
        if SciMLBase.isinplace(op)
            res = zero(op.output_cache)
            op.vjp_op(res, v, u, p)
            return res
        end
        return op.vjp_op(v, u, p)
    else
        if SciMLBase.isinplace(op)
            res = zero(op.output_cache)
            op.jvp_op(res, v, u, p)
            return res
        end
        return op.jvp_op(v, u, p)
    end
end

function (op::JacobianOperator)(::Number, ::Number, _, __)
    error("Inplace Jacobian Operator not possible for scalars.")
end

function (op::JacobianOperator)(Jv, v, u, p)
    if op.mode isa VJP
        if SciMLBase.isinplace(op)
            op.vjp_op(Jv, v, u, p)
        else
            copyto!(Jv, op.vjp_op(v, u, p))
        end
    else
        if SciMLBase.isinplace(op)
            op.jvp_op(Jv, v, u, p)
        else
            copyto!(Jv, op.jvp_op(v, u, p))
        end
    end
    return Jv
end

"""
    VecJacOperator(args...; autodiff = nothing, kwargs...)

Constructs a [`JacobianOperator`](@ref) which only provides the VJP using the
`vjp_autodiff = autodiff`.
"""
function VecJacOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_jvp = True, vjp_autodiff = autodiff)'
end

"""
    JacVecOperator(args...; autodiff = nothing, kwargs...)

Constructs a [`JacobianOperator`](@ref) which only provides the JVP using the
`jvp_autodiff = autodiff`.
"""
function JacVecOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_vjp = True, jvp_autodiff = autodiff)
end

"""
    StatefulJacobianOperator(jac_op::JacobianOperator, u, p)

Wrapper over a [`JacobianOperator`](@ref) which stores the input `u` and `p` and defines
`mul!` and `*` for computing VJPs and JVPs.
"""
@concrete struct StatefulJacobianOperator{M <: AbstractMode, T} <:
                 AbstractJacobianOperator{T}
    mode::M
    jac_op <: JacobianOperator
    u
    p

    function StatefulJacobianOperator(jac_op::JacobianOperator, u, p)
        return new{
            typeof(jac_op.mode), eltype(jac_op), typeof(jac_op), typeof(u), typeof(p)}(
            jac_op.mode, jac_op, u, p)
    end
end

Base.size(J::StatefulJacobianOperator) = size(J.jac_op)
Base.size(J::StatefulJacobianOperator, d::Integer) = size(J.jac_op, d)

for op in (:adjoint, :transpose)
    @eval function Base.$(op)(operator::StatefulJacobianOperator)
        return StatefulJacobianOperator($(op)(operator.jac_op), operator.u, operator.p)
    end
end

Base.:*(J::StatefulJacobianOperator, v::AbstractArray) = J.jac_op(v, J.u, J.p)

function LinearAlgebra.mul!(
        Jv::AbstractArray, J::StatefulJacobianOperator, v::AbstractArray)
    J.jac_op(Jv, v, J.u, J.p)
    return Jv
end

"""
    StatefulJacobianNormalFormOperator(vjp_operator, jvp_operator, cache)

This constructs a Normal Form Jacobian Operator, i.e. it constructs the operator
corresponding to `JᵀJ` where `J` is the Jacobian Operator. This is not meant to be directly
constructed, rather it is constructed with `*` on two [`StatefulJacobianOperator`](@ref)s.
"""
@concrete mutable struct StatefulJacobianNormalFormOperator{T} <:
                         AbstractJacobianOperator{T}
    vjp_operator <: StatefulJacobianOperator{VJP}
    jvp_operator <: StatefulJacobianOperator{JVP}
    cache
end

function Base.size(J::StatefulJacobianNormalFormOperator)
    return size(J.vjp_operator, 1), size(J.jvp_operator, 2)
end

function Base.:*(J1::StatefulJacobianOperator{VJP}, J2::StatefulJacobianOperator{JVP})
    cache = J2 * J2.jac_op.input_cache
    T = promote_type(eltype(J1), eltype(J2))
    return StatefulJacobianNormalFormOperator{T}(J1, J2, cache)
end

function LinearAlgebra.mul!(C::StatefulJacobianNormalFormOperator,
        A::StatefulJacobianOperator{VJP}, B::StatefulJacobianOperator{JVP})
    C.vjp_operator = A
    C.jvp_operator = B
    return C
end

function Base.:*(JᵀJ::StatefulJacobianNormalFormOperator, x::AbstractArray)
    return JᵀJ.vjp_operator * (JᵀJ.jvp_operator * x)
end

function LinearAlgebra.mul!(
        JᵀJx::AbstractArray, JᵀJ::StatefulJacobianNormalFormOperator, x::AbstractArray)
    mul!(JᵀJ.cache, JᵀJ.jvp_operator, x)
    mul!(JᵀJx, JᵀJ.vjp_operator, JᵀJ.cache)
    return JᵀJx
end

# Helper Functions
prepare_vjp(::Val{true}, args...; kwargs...) = nothing

function prepare_vjp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    return prepare_scalar_op(Val(false), prob, f, u, fu; autodiff)
end

function prepare_vjp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u, fu; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp

    if autodiff === nothing && SciMLBase.has_jac(f)
        if SciMLBase.isinplace(f)
            jac_cache = similar(u, eltype(fu), length(fu), length(u))
            return @closure (vJ, v, u, p) -> begin
                f.jac(jac_cache, u, p)
                mul!(vec(vJ), jac_cache', vec(v))
                return
            end
            return vjp_op, vjp_extras
        else
            return @closure (v, u, p) -> reshape(f.jac(u, p)' * vec(v), size(u))
        end
    end

    @assert autodiff!==nothing "`vjp_autodiff` must be provided if `f` doesn't have \
                                analytic `vjp` or `jac`."
    # TODO: Once DI supports const params we can use `p`
    fₚ = SciMLBase.JacobianWrapper{SciMLBase.isinplace(f)}(f, prob.p)
    if SciMLBase.isinplace(f)
        fu_cache = copy(fu)
        v_fake = copy(fu)
        di_extras = DI.prepare_pullback(fₚ, fu_cache, autodiff, u, v_fake)
        return @closure (vJ, v, u, p) -> begin
            DI.pullback!(fₚ, fu_cache, reshape(vJ, size(u)), autodiff, u, v, di_extras)
        end
    else
        di_extras = DI.prepare_pullback(fₚ, autodiff, u, fu)
        return @closure (v, u, p) -> DI.pullback(fₚ, autodiff, u, v, di_extras)
    end
end

prepare_jvp(skip::Val{true}, args...; kwargs...) = nothing

function prepare_jvp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    return prepare_scalar_op(Val(false), prob, f, u, fu; autodiff)
end

function prepare_jvp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u, fu; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp

    if autodiff === nothing && SciMLBase.has_jac(f)
        if SciMLBase.isinplace(f)
            jac_cache = similar(u, eltype(fu), length(fu), length(u))
            return @closure (Jv, v, u, p) -> begin
                f.jac(jac_cache, u, p)
                mul!(vec(Jv), jac_cache, vec(v))
                return
            end
        else
            return @closure (v, u, p, _) -> reshape(f.jac(u, p) * vec(v), size(u))
        end
    end

    @assert autodiff!==nothing "`jvp_autodiff` must be provided if `f` doesn't have \
                                analytic `vjp` or `jac`."
    # TODO: Once DI supports const params we can use `p`
    fₚ = SciMLBase.JacobianWrapper{SciMLBase.isinplace(f)}(f, prob.p)
    if SciMLBase.isinplace(f)
        fu_cache = copy(fu)
        di_extras = DI.prepare_pushforward(fₚ, fu_cache, autodiff, u, u)
        return @closure (Jv, v, u, p) -> begin
            DI.pushforward!(
                fₚ, fu_cache, reshape(Jv, size(fu_cache)), autodiff, u, v, di_extras)
            return
        end
    else
        di_extras = DI.prepare_pushforward(fₚ, autodiff, u, u)
        return @closure (v, u, p) -> DI.pushforward(fₚ, autodiff, u, v, di_extras)
    end
end

function prepare_scalar_op(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp
    SciMLBase.has_jvp(f) && return f.jvp
    SciMLBase.has_jac(f) && return @closure((v, u, p)->f.jac(u, p) * v)

    @assert autodiff!==nothing "`autodiff` must be provided if `f` doesn't have \
                                analytic `vjp` or `jvp` or `jac`."
    # TODO: Once DI supports const params we can use `p`
    fₚ = Base.Fix2(f, prob.p)
    di_extras = DI.prepare_derivative(fₚ, autodiff, u)
    return @closure (v, u, p) -> DI.derivative(fₚ, autodiff, u, di_extras) * v
end

export JacobianOperator, VecJacOperator, JacVecOperator
export StatefulJacobianOperator
export StatefulJacobianNormalFormOperator

end
