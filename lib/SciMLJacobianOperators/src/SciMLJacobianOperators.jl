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

abstract type AbstractMode end

struct VJP <: AbstractMode end
struct JVP <: AbstractMode end

flip_mode(::VJP) = JVP()
flip_mode(::JVP) = VJP()

@concrete struct JacobianOperator{iip, T <: Real} <: AbstractSciMLOperator{T}
    mode <: AbstractMode

    jvp_op
    vjp_op

    size
    jvp_extras
    vjp_extras
end

function ConstructionBase.constructorof(::Type{<:JacobianOperator{iip, T}}) where {iip, T}
    return JacobianOperator{iip, T}
end

Base.size(J::JacobianOperator) = J.size
Base.size(J::JacobianOperator, d::Integer) = J.size[d]

for op in (:adjoint, :transpose)
    @eval function Base.$(op)(operator::JacobianOperator)
        @set! operator.mode = flip_mode(operator.mode)
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
    fₚ = SciMLBase.JacobianWrapper{iip}(f, prob.p)

    vjp_op, vjp_extras = prepare_vjp(skip_vjp, prob, f, u, fu; autodiff = vjp_autodiff)
    jvp_op, jvp_extras = prepare_jvp(skip_jvp, prob, f, u, fu; autodiff = jvp_autodiff)

    return JacobianOperator{iip, T}(
        JVP(), jvp_op, vjp_op, (length(fu), length(u)), jvp_extras, vjp_extras)
end

prepare_vjp(::Val{true}, args...; kwargs...) = nothing, nothing

function prepare_vjp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    return prepare_scalar_op(Val(false), prob, f, u, fu; autodiff)
end

function prepare_vjp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u, fu; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp, nothing

    if autodiff === nothing && SciMLBase.has_jac(f)
        if SciMLBase.isinplace(f)
            vjp_extras = (; jac_cache = similar(u, eltype(fu), length(fu), length(u)))
            vjp_op = @closure (vJ, v, u, p, extras) -> begin
                f.jac(extras.jac_cache, u, p)
                mul!(vec(vJ), extras.jac_cache', vec(v))
                return
            end
            return vjp_op, vjp_extras
        else
            vjp_op = @closure (v, u, p, _) -> reshape(f.jac(u, p)' * vec(v), size(u))
            return vjp_op, nothing
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
        vjp_op = @closure (vJ, v, u, p, extras) -> begin
            DI.pullback!(
                fₚ, extras.fu_cache, reshape(vJ, size(u)), autodiff, u, v, extras.di_extras)
        end
        return vjp_op, (; di_extras, fu_cache)
    else
        di_extras = DI.prepare_pullback(f, autodiff, u, fu)
        vjp_op = @closure (v, u, p, extras) -> begin
            return DI.pullback(f, autodiff, u, v, extras.di_extras)
        end
        return vjp_op, (; di_extras)
    end
end

prepare_jvp(skip::Val{true}, args...; kwargs...) = nothing, nothing

function prepare_jvp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    return prepare_scalar_op(Val(false), prob, f, u, fu; autodiff)
end

function prepare_jvp(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u, fu; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp, nothing

    if autodiff === nothing && SciMLBase.has_jac(f)
        if SciMLBase.isinplace(f)
            jvp_extras = (; jac_cache = similar(u, eltype(fu), length(fu), length(u)))
            jvp_op = @closure (Jv, v, u, p, extras) -> begin
                f.jac(extras.jac_cache, u, p)
                mul!(vec(Jv), extras.jac_cache, vec(v))
                return
            end
            return jvp_op, jvp_extras
        else
            jvp_op = @closure (v, u, p, _) -> reshape(f.jac(u, p) * vec(v), size(u))
            return jvp_op, nothing
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
        jvp_op = @closure (Jv, v, u, p, extras) -> begin
            DI.pushforward!(fₚ, extras.fu_cache, reshape(Jv, size(extras.fu_cache)),
                autodiff, u, v, extras.di_extras)
        end
        return jvp_op, (; di_extras, fu_cache)
    else
        di_extras = DI.prepare_pushforward(f, autodiff, u, u)
        jvp_op = @closure (v, u, p, extras) -> begin
            return DI.pushforward(f, autodiff, u, v, extras.di_extras)
        end
        return jvp_op, (; di_extras)
    end
end

function prepare_scalar_op(::Val{false}, prob::AbstractNonlinearProblem,
        f::AbstractNonlinearFunction, u::Number, fu::Number; autodiff = nothing)
    SciMLBase.has_vjp(f) && return f.vjp, nothing
    SciMLBase.has_jvp(f) && return f.jvp, nothing
    SciMLBase.has_jac(f) && return @closure((v, u, p, _)->f.jac(u, p) * v), nothing

    @assert autodiff!==nothing "`autodiff` must be provided if `f` doesn't have \
                                analytic `vjp` or `jvp` or `jac`."
    # TODO: Once DI supports const params we can use `p`
    fₚ = Base.Fix2(f, prob.p)
    di_extras = DI.prepare_derivative(fₚ, autodiff, u)
    op = @closure (v, u, p, extras) -> begin
        return DI.derivative(fₚ, autodiff, u, extras.di_extras) * v
    end
    return op, (; di_extras)
end

function VecJacOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_jvp = True, vjp_autodiff = autodiff)'
end

function JacVecOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_vjp = True, jvp_autodiff = autodiff)
end

export JacobianOperator, VecJacOperator, JacVecOperator

end
