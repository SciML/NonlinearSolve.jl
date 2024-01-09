# We want a general form of this in SciMLOperators. However, we use this extensively and we
# can have a custom implementation here till
# https://github.com/SciML/SciMLOperators.jl/issues/223 is resolved.
"""
    JacobianOperator{vjp, iip, T} <: AbstractNonlinearSolveOperator{T}

A Jacobian Operator Provides both JVP and VJP without materializing either (if possible).

This is an internal operator, and is not guaranteed to have a stable API. It might even be
moved out of NonlinearSolve.jl in the future, without a deprecation cycle. Usage of this
outside NonlinearSolve.jl (by everyone except Avik) is strictly prohibited.

`T` denotes if the Jacobian is transposed or not. `T = true` means that the Jacobian is
transposed, and `T = false` means that the Jacobian is not transposed.

### Constructor

```julia
JacobianOperator(prob::AbstractNonlinearProblem, fu, u; jvp_autodiff = nothing,
    vjp_autodiff = nothing, skip_vjp::Val{NoVJP} = False,
    skip_jvp::Val{NoJVP} = False) where {NoVJP, NoJVP}
```

Shorthand constructors are also available:

```julia
VecJacOperator(args...; autodiff = nothing, kwargs...)
JacVecOperator(args...; autodiff = nothing, kwargs...)
```
"""
@concrete struct JacobianOperator{vjp, iip, T} <: AbstractNonlinearSolveOperator{T}
    jvp_op
    vjp_op

    input_cache
    output_cache
end

Base.size(J::JacobianOperator) = prod(size(J.output_cache)), prod(size(J.input_cache))
function Base.size(J::JacobianOperator, d::Integer)
    if d == 1
        return prod(size(J.output_cache))
    elseif d == 2
        return prod(size(J.input_cache))
    else
        error("Invalid dimension $d for JacobianOperator")
    end
end

for op in (:adjoint, :transpose)
    @eval function Base.$(op)(operator::JacobianOperator{vjp, iip, T}) where {vjp, iip, T}
        return JacobianOperator{!vjp, iip, T}(operator.jvp_op, operator.vjp_op,
            operator.output_cache, operator.input_cache)
    end
end

function JacobianOperator(prob::AbstractNonlinearProblem, fu, u; jvp_autodiff = nothing,
        vjp_autodiff = nothing, skip_vjp::Val{NoVJP} = False,
        skip_jvp::Val{NoJVP} = False) where {NoVJP, NoJVP}
    f = prob.f
    iip = isinplace(prob)
    uf = JacobianWrapper{iip}(f, prob.p)

    vjp_op = if NoVJP
        nothing
    elseif SciMLBase.has_vjp(f)
        f.vjp
    else
        vjp_autodiff = __get_nonsparse_ad(get_concrete_reverse_ad(vjp_autodiff,
            prob, False))
        if vjp_autodiff isa AutoZygote
            iip && error("`AutoZygote` cannot handle inplace problems.")
            @closure (v, u, p) -> auto_vecjac(uf, u, v)
        elseif vjp_autodiff isa AutoFiniteDiff
            if iip
                cache1 = similar(fu)
                cache2 = similar(fu)
                @closure (Jv, v, u, p) -> num_vecjac!(Jv, uf, u, v, cache1, cache2)
            else
                @closure (v, u, p) -> num_vecjac(uf, __mutable(u), v)
            end
        else
            error("`vjp_autodiff` = `$(typeof(vjp_autodiff))` is not supported in \
                   JacobianOperator.")
        end
    end

    jvp_op = if NoJVP
        nothing
    elseif SciMLBase.has_jvp(f)
        f.jvp
    else
        jvp_autodiff = __get_nonsparse_ad(get_concrete_forward_ad(jvp_autodiff,
            prob, False))
        if jvp_autodiff isa AutoForwardDiff || jvp_autodiff isa AutoPolyesterForwardDiff
            if iip
                # FIXME: Technically we should propagate the tag but ignoring that for now
                cache1 = Dual{
                    typeof(ForwardDiff.Tag(NonlinearSolveTag(), eltype(u))), eltype(u), 1,
                }.(similar(u), ForwardDiff.Partials.(tuple.(u)))
                cache2 = Dual{
                    typeof(ForwardDiff.Tag(NonlinearSolveTag(), eltype(fu))), eltype(fu), 1,
                }.(similar(fu), ForwardDiff.Partials.(tuple.(fu)))
                @closure (Jv, v, u, p) -> auto_jacvec!(Jv, uf, u, v, cache1, cache2)
            else
                @closure (v, u, p) -> auto_jacvec(uf, u, v)
            end
        elseif jvp_autodiff isa AutoFiniteDiff
            if iip
                cache1 = similar(fu)
                cache2 = similar(u)
                @closure (Jv, v, u, p) -> num_jacvec!(Jv, uf, u, v, cache1, cache2)
            else
                @closure (v, u, p) -> num_jacvec(uf, u, v)
            end
        else
            error("`jvp_autodiff` = `$(typeof(jvp_autodiff))` is not supported in \
                   JacobianOperator.")
        end
    end

    return JacobianOperator{false, iip, promote_type(eltype(fu), eltype(u))}(jvp_op, vjp_op,
        u, fu)
end

function VecJacOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_jvp = True, vjp_autodiff = autodiff)'
end
function JacVecOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_vjp = True, jvp_autodiff = autodiff)
end

function (op::JacobianOperator{vjp, iip})(v, u, p) where {vjp, iip}
    if vjp
        if iip
            res = similar(op.output_cache)
            op.vjp_op(res, v, u, p)
            return res
        else
            return op.vjp_op(v, u, p)
        end
    else
        if iip
            res = similar(op.output_cache)
            op.jvp_op(res, v, u, p)
            return res
        else
            return op.jvp_op(v, u, p)
        end
    end
end

# Prevent Ambiguity
function (op::JacobianOperator{vjp, iip})(Jv::Number, v::Number, u, p) where {vjp, iip}
    error("Inplace Jacobian Operator not possible for scalars.")
end

function (op::JacobianOperator{vjp, iip})(Jv, v, u, p) where {vjp, iip}
    if vjp
        if iip
            op.vjp_op(Jv, v, u, p)
        else
            copyto!(Jv, op.vjp_op(v, u, p))
        end
    else
        if iip
            op.jvp_op(Jv, v, u, p)
        else
            copyto!(Jv, op.jvp_op(v, u, p))
        end
    end
    return Jv
end

@concrete struct StatefulJacobianOperator{vjp, iip, T,
    J <: JacobianOperator{vjp, iip, T}} <: AbstractNonlinearSolveOperator{T}
    jac_op::J
    u
    p
end

Base.size(J::StatefulJacobianOperator) = size(J.jac_op)
Base.size(J::StatefulJacobianOperator, d::Integer) = size(J.jac_op, d)

for op in (:adjoint, :transpose)
    @eval function Base.$op(operator::StatefulJacobianOperator)
        return StatefulJacobianOperator($(op)(operator.jac_op), operator.u, operator.p)
    end
end

Base.:*(J::StatefulJacobianOperator, v::AbstractArray) = J.jac_op(v, J.u, J.p)

function LinearAlgebra.mul!(Jv::AbstractArray, J::StatefulJacobianOperator,
        v::AbstractArray)
    J.jac_op(Jv, v, J.u, J.p)
    return Jv
end

@concrete mutable struct StatefulJacobianNormalFormOperator{T} <:
                         AbstractNonlinearSolveOperator{T}
    vjp_operator
    jvp_operator
    cache
end

function Base.size(J::StatefulJacobianNormalFormOperator)
    return size(J.vjp_operator, 1), size(J.jvp_operator, 2)
end

function Base.:*(J1::StatefulJacobianOperator{true}, J2::StatefulJacobianOperator{false})
    cache = J2 * J2.jac_op.input_cache
    T = promote_type(eltype(J1), eltype(J2))
    return StatefulJacobianNormalFormOperator{T}(J1, J2, cache)
end

function LinearAlgebra.mul!(C::StatefulJacobianNormalFormOperator,
        A::StatefulJacobianOperator{true}, B::StatefulJacobianOperator{false})
    C.vjp_operator = A
    C.jvp_operator = B
    return C
end

function Base.:*(JᵀJ::StatefulJacobianNormalFormOperator, x::AbstractArray)
    return JᵀJ.vjp_operator * (JᵀJ.jvp_operator * x)
end

function LinearAlgebra.mul!(JᵀJx::AbstractArray, JᵀJ::StatefulJacobianNormalFormOperator,
        x::AbstractArray)
    mul!(JᵀJ.cache, JᵀJ.jvp_operator, x)
    mul!(JᵀJx, JᵀJ.vjp_operator, JᵀJ.cache)
    return JᵀJx
end
