module Utils

using ArrayInterface: ArrayInterface
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, I, diagind
using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem,
                          AbstractNonlinearTerminationMode,
                          AbstractSafeNonlinearTerminationMode,
                          AbstractSafeBestNonlinearTerminationMode
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NonlinearLeastSquaresProblem,
                 NonlinearProblem, NonlinearFunction, ReturnCode
using StaticArraysCore: StaticArray, SArray, SMatrix, SVector

const DI = DifferentiationInterface

const safe_similar = NonlinearSolveBase.Utils.safe_similar

pickchunksize(n::Int) = min(n, 12)

can_dual(::Type{<:Real}) = true
can_dual(::Type) = false

maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
function maybe_unaliased(x::T, alias::Bool) where {T <: AbstractArray}
    (alias || !ArrayInterface.can_setindex(T)) && return x
    return copy(x)
end

# NOTE: This doesn't initialize the `f(x)` but just returns a buffer of the same size
function get_fx(prob::NonlinearLeastSquaresProblem, x)
    if SciMLBase.isinplace(prob) && prob.f.resid_prototype === nothing
        error("Inplace NonlinearLeastSquaresProblem requires a `resid_prototype` to be \
               specified.")
    end
    return get_fx(prob.f, x, prob.p)
end
function get_fx(prob::Union{ImmutableNonlinearProblem, NonlinearProblem}, x)
    return get_fx(prob.f, x, prob.p)
end
function get_fx(f::NonlinearFunction, x, p)
    if SciMLBase.isinplace(f)
        f.resid_prototype === nothing || return eltype(x).(f.resid_prototype)
        return safe_similar(x)
    end
    return f(x, p)
end

function eval_f(prob, fx, x)
    SciMLBase.isinplace(prob) || return prob.f(x, prob.p)
    prob.f(fx, x, prob.p)
    return fx
end

function fixed_parameter_function(prob::AbstractNonlinearProblem)
    SciMLBase.isinplace(prob) && return @closure (du, u) -> prob.f(du, u, prob.p)
    return Base.Fix2(prob.f, prob.p)
end

function identity_jacobian(u::Number, fu::Number, α = true)
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
function identity_jacobian(u, fu, α = true)
    J = safe_similar(u, promote_type(eltype(u), eltype(fu)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= eltype(J)(α)
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
        cache, fx, x, xo, prob, ::AbstractSafeBestNonlinearTerminationMode)
    if cache(fx, x, xo)
        x = cache.u
        if SciMLBase.isinplace(prob)
            prob.f(fx, x, prob.p)
        else
            fx = prob.f(x, prob.p)
        end
        return true, cache.retcode, fx, x
    end
    return false, ReturnCode.Default, fx, x
end

restructure(y, x) = ArrayInterface.restructure(y, x)
restructure(::Number, x::Number) = x

safe_vec(x::AbstractArray) = vec(x)
safe_vec(x::Number) = x

abstract type AbstractJacobianMode end

struct AnalyticJacobian <: AbstractJacobianMode end
@concrete struct DIExtras <: AbstractJacobianMode
    prep
end
struct DINoPreparation <: AbstractJacobianMode end

# While we could run prep in other cases, we don't since we need it completely
# non-allocating for running inside GPU kernels
function prepare_jacobian(prob, autodiff, _, x::Number)
    if SciMLBase.has_jac(prob.f) || SciMLBase.has_vjp(prob.f) || SciMLBase.has_jvp(prob.f)
        return AnalyticJacobian()
    end
    # return DI.prepare_derivative(prob.f, autodiff, x, Constant(prob.p))
    return DINoPreparation()
end
function prepare_jacobian(prob, autodiff, fx, x)
    SciMLBase.has_jac(prob.f) && return AnalyticJacobian()
    if SciMLBase.isinplace(prob.f)
        return DIExtras(DI.prepare_jacobian(prob.f, fx, autodiff, x, Constant(prob.p)))
    else
        x isa SArray && return DINoPreparation()
        return DI.prepare_jacobian(prob.f, autodiff, x, Constant(prob.p))
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
            J = safe_similar(fx, length(fx), length(x))
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
        DI.jacobian!(prob.f, J, fx, extras.prep, autodiff, x, Constant(prob.p))
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
    @assert !SciMLBase.isinplace(prob.f) "This shouldn't happen. Open an issue."
    J === nothing && return DI.jacobian(prob.f, autodiff, x, Constant(prob.p))
    if ArrayInterface.can_setindex(J)
        DI.jacobian!(prob.f, J, autodiff, x, Constant(prob.p))
    else
        J = DI.jacobian(prob.f, autodiff, x, Constant(prob.p))
    end
    return J
end

function compute_jacobian_and_hessian(autodiff, prob, _, x::Number)
    H = DI.second_derivative(prob.f, autodiff, x, Constant(prob.p))
    fx, J = DI.value_and_derivative(prob.f, autodiff, x, Constant(prob.p))
    return fx, J, H
end
function compute_jacobian_and_hessian(autodiff, prob, fx, x)
    if SciMLBase.isinplace(prob)
        jac_fn = @closure (u, p) -> begin
            du = safe_similar(fx, promote_type(eltype(fx), eltype(u)))
            return DI.jacobian(prob.f, du, autodiff, u, Constant(p))
        end
        J, H = DI.value_and_jacobian(jac_fn, autodiff, x, Constant(prob.p))
        fx = Utils.eval_f(prob, fx, x)
        return fx, J, H
    else
        jac_fn = @closure (u, p) -> DI.jacobian(prob.f, autodiff, u, Constant(p))
        J, H = DI.value_and_jacobian(jac_fn, autodiff, x, Constant(prob.p))
        fx = Utils.eval_f(prob, fx, x)
        return fx, J, H
    end
end

end
