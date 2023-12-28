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

Base.size(J::JacobianOperator) = (prod(size(J.output_cache)), prod(size(J.input_cache)))
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
    @eval function Base.$op(op::JacobianOperator{vjp, iip, T}) where {vjp, iip, T}
        return JacobianOperator{!vjp, iip, T}(op.jvp_op, op.vjp_op, op.output_cache,
            op.input_cache)
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
                cache2 = similar(u)
                @closure (Jv, v, u, p) -> num_vecjac!(Jv, uf, u, v, cache1, cache2)
            else
                @closure (v, u, p) -> num_vecjac(uf, u, v)
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
    op = JacobianOperator(args...; kwargs..., skip_jvp = True, vjp_autodiff = autodiff)'
    return op
end
function JacVecOperator(args...; autodiff = nothing, kwargs...)
    return JacobianOperator(args...; kwargs..., skip_vjp = True, jvp_autodiff = autodiff)
end

function (op::JacobianOperator{vjp, iip})(v, u, p) where {vjp, iip}
    if vjp
        if iip
            res = similar(J.input_cache)
            op.vjp_op(res, v, u, p)
            return res
        else
            return op.vjp_op(v, u, p)
        end
    else
        if iip
            res = similar(J.output_cache)
            op.jvp_op(res, v, u, p)
            return res
        else
            return op.jvp_op(v, u, p)
        end
    end
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

Base.:*(J::StatefulJacobianOperator, v::AbstractArray) = J.jac_op(v, J.u, J.p)
function LinearAlgebra.mul!(Jv::AbstractArray, J::StatefulJacobianOperator,
        v::AbstractArray)
    J.jac_op(Jv, v, J.u, J.p)
    return Jv
end

# TODO: Define JacobianOperatorᵀ * JacobianOperator for Normal Form Krylov Solvers, even
#       though in these cases solvers like LSMR should be used.
