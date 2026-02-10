module Utils

using ArrayInterface: ArrayInterface
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, I, diagind
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearTerminationMode,
    AbstractSafeNonlinearTerminationMode,
    AbstractSafeBestNonlinearTerminationMode
using SciMLBase: SciMLBase, ReturnCode
using StaticArraysCore: StaticArray, SArray, SMatrix, SVector

const DI = DifferentiationInterface
const NLBUtils = NonlinearSolveBase.Utils

# GPU-compatible helper to extract alias_u0 from NonlinearAliasSpecifier
@inline function get_alias_u0(alias::SciMLBase.NonlinearAliasSpecifier, fallback::Bool)
    return something(alias.alias_u0, fallback)
end

# GPU-compatible helper to check if fx should be cached
@generated function should_cache_fx(prob::SciMLBase.AbstractNonlinearProblem, f)
    iip = prob <: SciMLBase.AbstractNonlinearProblem{<:Any, true}
    return quote
        $iip && !SciMLBase.has_jac(f)
    end
end

function identity_jacobian(u::Number, fu::Number, α = true)
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
function identity_jacobian(u, fu, α = true)
    J = NLBUtils.safe_similar(u, promote_type(eltype(u), eltype(fu)), length(fu), length(u))
    fill!(J, false)
    if ArrayInterface.fast_scalar_indexing(J)
        @simd ivdep for i in axes(J, 1)
            @inbounds J[i, i] = α
        end
    else
        J[diagind(J)] .= α
    end
    return J
end
function identity_jacobian(u::StaticArray, fu, α = true)
    return SMatrix{length(fu), length(u), promote_type(eltype(fu), eltype(u))}(I * α)
end

identity_jacobian!!(J::Number) = one(J)
function identity_jacobian!!(J::AbstractVector)
    ArrayInterface.can_setindex(J) || return one.(J)
    fill!(J, true)
    return J
end
function identity_jacobian!!(J::AbstractMatrix)
    ArrayInterface.can_setindex(J) || return convert(typeof(J), I)
    J[diagind(J)] .= true
    return J
end
identity_jacobian!!(::SMatrix{S1, S2, T}) where {S1, S2, T} = SMatrix{S1, S2, T}(I)
identity_jacobian!!(::SVector{S1, T}) where {S1, T} = ones(SVector{S1, T})

# Termination Conditions
function check_termination(cache, fx, x, xo, prob)
    return check_termination(cache, fx, x, xo, prob, cache.mode)
end

function check_termination(cache, fx, x, xo, _, ::AbstractNonlinearTerminationMode)
    return cache(fx, x, xo), ReturnCode.Success, fx, x
end
function check_termination(cache, fx, x, xo, _, ::AbstractSafeNonlinearTerminationMode)
    return cache(fx, x, xo), cache.retcode, fx, x
end
function check_termination(
        cache, fx, x, xo, prob, ::AbstractSafeBestNonlinearTerminationMode
    )
    if cache(fx, x, xo)
        x = cache.u
        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        return true, cache.retcode, fx, x
    end
    return false, ReturnCode.Default, fx, x
end

abstract type AbstractJacobianMode end

struct AnalyticJacobian <: AbstractJacobianMode end
struct DIExtras{P} <: AbstractJacobianMode
    prep::P
end
struct DINoPreparation <: AbstractJacobianMode end

# While we could run prep in other cases, we don't since we need it completely
# non-allocating for running inside GPU kernels
function prepare_jacobian(prob, autodiff, _, x::Number)
    if SciMLBase.has_jac(prob.f) || SciMLBase.has_vjp(prob.f) || SciMLBase.has_jvp(prob.f)
        return AnalyticJacobian()
    end
    return DINoPreparation()
end

@generated function prepare_jacobian(prob, autodiff, fx, x)
    iip = prob <: SciMLBase.AbstractNonlinearProblem{<:Any, true}
    if iip
        return quote
            SciMLBase.has_jac(prob.f) && return AnalyticJacobian()
            return DIExtras(
                DI.prepare_jacobian(
                    prob.f, fx, autodiff, x, Constant(prob.p), strict = Val(false)
                )
            )
        end
    else
        return quote
            SciMLBase.has_jac(prob.f) && return AnalyticJacobian()
            x isa SArray && return DINoPreparation()
            return DIExtras(
                DI.prepare_jacobian(
                    prob.f, autodiff, x, Constant(prob.p), strict = Val(false)
                )
            )
        end
    end
end

function compute_jacobian!!(_, prob, autodiff, fx, x::Number, ::AnalyticJacobian)
    if SciMLBase.has_jac(prob.f)
        return prob.f.jac(x, prob.p)
    elseif SciMLBase.has_vjp(prob.f)
        return prob.f.vjp(one(x), x, prob.p)
    elseif SciMLBase.has_jvp(prob.f)
        return prob.f.jvp(one(x), x, prob.p)
    end
end
function compute_jacobian!!(_, prob, autodiff, fx, x::Number, extras::DIExtras)
    return DI.derivative(prob.f, extras.prep, autodiff, x, Constant(prob.p))
end
function compute_jacobian!!(_, prob, autodiff, fx, x::Number, ::DINoPreparation)
    return DI.derivative(prob.f, autodiff, x, Constant(prob.p))
end

function compute_jacobian!!(J, prob, autodiff, fx, x, ::AnalyticJacobian)
    if J === nothing
        if SciMLBase.isinplace(prob.f)
            J = NLBUtils.safe_similar(fx, length(fx), length(x))
            prob.f.jac(J, x, prob.p)
            return J
        else
            return prob.f.jac(x, prob.p)
        end
    end
    if SciMLBase.isinplace(prob.f)
        prob.f.jac(J, x, prob.p)
        return J
    else
        return prob.f.jac(x, prob.p)
    end
end

function compute_jacobian!!(J, prob, autodiff, fx, x, extras::DIExtras)
    if J === nothing
        if SciMLBase.isinplace(prob.f)
            return DI.jacobian(prob.f, fx, extras.prep, autodiff, x, Constant(prob.p))
        else
            return DI.jacobian(prob.f, extras.prep, autodiff, x, Constant(prob.p))
        end
    end
    if SciMLBase.isinplace(prob.f)
        DI.jacobian!(prob.f, fx, J, extras.prep, autodiff, x, Constant(prob.p))
    else
        if ArrayInterface.can_setindex(J)
            DI.jacobian!(prob.f, J, extras.prep, autodiff, x, Constant(prob.p))
        else
            J = DI.jacobian(prob.f, extras.prep, autodiff, x, Constant(prob.p))
        end
    end
    return J
end
function compute_jacobian!!(J, prob, autodiff, fx, x, ::DINoPreparation)
    # Assertion removed for GPU compatibility - DINoPreparation only used for out-of-place
    # @assert !SciMLBase.isinplace(prob.f) "This shouldn't happen. Open an issue."
    J === nothing && return DI.jacobian(prob.f, autodiff, x, Constant(prob.p))
    if ArrayInterface.can_setindex(J)
        DI.jacobian!(prob.f, J, autodiff, x, Constant(prob.p))
    else
        J = DI.jacobian(prob.f, autodiff, x, Constant(prob.p))
    end
    return J
end

function compute_hvvp(prob, autodiff, _, x::Number, dir::Number)
    H = DI.second_derivative(prob.f, autodiff, x, Constant(prob.p))
    return H * dir
end

@generated function compute_hvvp(prob, autodiff, fx, x, dir)
    iip = prob <: SciMLBase.AbstractNonlinearProblem{<:Any, true}
    if iip
        return quote
            jvp_fn = @closure (u, p) -> begin
                du = NLBUtils.safe_similar(fx, promote_type(eltype(fx), eltype(u)))
                return only(DI.pushforward(prob.f, du, autodiff, u, (dir,), Constant(p)))
            end
            return only(DI.pushforward(jvp_fn, autodiff, x, (dir,), Constant(prob.p)))
        end
    else
        return quote
            jvp_fn = @closure (u, p) -> only(DI.pushforward(prob.f, autodiff, u, (dir,), Constant(p)))
            return only(DI.pushforward(jvp_fn, autodiff, x, (dir,), Constant(prob.p)))
        end
    end
end

function nonlinear_solution_new_alg(
        sol::SciMLBase.NonlinearSolution{T, N, uType, R, P, A, O, uType2, S, Tr}, alg
    ) where {T, N, uType, R, P, A, O, uType2, S, Tr}
    return SciMLBase.NonlinearSolution{T, N, uType, R, P, typeof(alg), O, uType2, S, Tr}(
        sol.u, sol.resid, sol.prob, alg, sol.retcode, sol.original, sol.left, sol.right,
        sol.stats, sol.trace
    )
end

end
