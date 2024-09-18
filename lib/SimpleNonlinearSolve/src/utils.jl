module Utils

using ADTypes: AbstractADType, AutoForwardDiff, AutoFiniteDiff, AutoPolyesterForwardDiff
using ArrayInterface: ArrayInterface
using DifferentiationInterface: DifferentiationInterface
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

function get_concrete_autodiff(_, ad::AbstractADType)
    DI.check_available(ad) && return ad
    error("AD Backend $(ad) is not available. This could be because you haven't loaded the \
           actual backend (See [Differentiation Interface Docs](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/) \
           for more details) or the backend might not be supported by DifferentiationInterface.jl.")
end
function get_concrete_autodiff(
        prob, ad::Union{AutoForwardDiff{nothing}, AutoPolyesterForwardDiff{nothing}})
    return get_concrete_autodiff(prob,
        ArrayInterface.parameterless_type(ad)(;
            chunksize = pickchunksize(length(prob.u0)), ad.tag))
end
function get_concrete_autodiff(prob, ::Nothing)
    if can_dual(eltype(prob.u0)) && DI.check_available(AutoForwardDiff())
        return AutoForwardDiff(; chunksize = pickchunksize(length(prob.u0)))
    end
    DI.check_available(AutoFiniteDiff()) && return AutoFiniteDiff()
    error("Default AD backends are not available. Please load either FiniteDiff or \
           ForwardDiff for default AD selection to work. Else provide a specific AD \
           backend (instead of `nothing`) to the solver.")
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
    J = safe_similar(u, promote_type(eltype(u), eltype(fu)))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= eltype(J)(α)
    return J
end
function identity_jacobian(u::StaticArray, fu, α = true)
    return SMatrix{length(fu), length(u), eltype(u)}(I * α)
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
function check_termination(cache, fx, x, xo, prob, ::AbstractSafeBestNonlinearTerminationMode)
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

end
