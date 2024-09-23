module SciMLJacobianOperators

using ADTypes: ADTypes
using ConcreteStructs: @concrete
using ConstructionBase: ConstructionBase
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
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

    output_cache = iip ? similar(fu, T) : nothing
    input_cache = iip ? similar(u, T) : nothing

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
            return
        end
        copyto!(Jv, op.vjp_op(v, u, p))
        return
    else
        if SciMLBase.isinplace(op)
            op.jvp_op(Jv, v, u, p)
            return
        end
        copyto!(Jv, op.jvp_op(v, u, p))
        return
    end
end

function VecJacOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_jvp = True, vjp_autodiff = autodiff)'
end

function JacVecOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_vjp = True, jvp_autodiff = autodiff)
end

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

    if ADTypes.mode(autodiff) isa ADTypes.ForwardMode
        @warn "AD Backend: $(autodiff) is a Forward Mode backend. Computing VJPs using \
               this will be slow!"
    end

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
        di_extras = DI.prepare_pullback(f, autodiff, u, fu)
        return @closure (v, u, p) -> DI.pullback(f, autodiff, u, v, di_extras)
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

    if ADTypes.mode(autodiff) isa ADTypes.ReverseMode
        @warn "AD Backend: $(autodiff) is a Reverse Mode backend. Computing JVPs using \
               this will be slow!"
    end

    # TODO: Once DI supports const params we can use `p`
    fₚ = SciMLBase.JacobianWrapper{SciMLBase.isinplace(f)}(f, prob.p)
    if SciMLBase.isinplace(f)
        fu_cache = copy(fu)
        di_extras = DI.prepare_pushforward(fₚ, fu_cache, autodiff, u, u)
        return @closure (Jv, v, u, p) -> begin
            DI.pushforward!(fₚ, fu_cache, reshape(Jv, size(fu_cache)), autodiff, u, v,
                di_extras)
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

end
