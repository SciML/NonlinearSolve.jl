module NonlinearSolveBaseForwardDiffExt

using ADTypes: ADTypes, AutoForwardDiff, AutoPolyesterForwardDiff
using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve, solve, solve!, init
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual, pickchunksize
using FunctionWrappers: FunctionWrappers
import FunctionWrappersWrappers
using SciMLBase: SciMLBase, AbstractNonlinearProblem, IntervalNonlinearProblem,
    NonlinearProblem, NonlinearLeastSquaresProblem, remake

using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem, Utils, InternalAPI,
    NonlinearSolvePolyAlgorithm, NonlinearSolveForwardDiffCache,
    NonlinearSolveTag, AutoSpecializeCallable, is_fw_wrapped

import NonlinearSolveBase: wrapfun_iip, standardize_forwarddiff_tag

const DI = DifferentiationInterface

# --- AutoSpecialize / norecompile infrastructure for ForwardDiff ---

const dualT = ForwardDiff.Dual{
    ForwardDiff.Tag{NonlinearSolveTag, Float64}, Float64, 1,
}
dualgen(::Type{T}) where {T} = ForwardDiff.Dual{
    ForwardDiff.Tag{NonlinearSolveTag, T}, T, 1,
}

# Fast-path dispatch for IIP calls with NonlinearSolveTag duals.
# These bypass the generic fallback path, calling directly into FunctionWrappersWrapper
# for zero-allocation dispatch.
@inline function (f::AutoSpecializeCallable)(
        du::Vector{dualT}, u::Vector{dualT}, p::Vector{Float64},
    )
    return f.fw(du, u, p)
end
@inline function (f::AutoSpecializeCallable)(
        du::Vector{dualT}, u::Vector{dualT}, p::SciMLBase.NullParameters,
    )
    return f.fw(du, u, p)
end
@inline function (f::AutoSpecializeCallable)(
        du::Vector{dualT}, u::Vector{dualT}, p::Vector{dualT},
    )
    return f.fw(du, u, p)
end
@inline function (f::AutoSpecializeCallable)(
        du::Vector{dualT}, u::Vector{Float64}, p::Vector{dualT},
    )
    return f.fw(du, u, p)
end
@inline function (f::AutoSpecializeCallable)(
        du::Vector{dualT}, u::Vector{dualT}, p::Float64,
    )
    return f.fw(du, u, p)
end

# Helper: build the canonical AutoForwardDiff for wrapped functions (chunksize=1 + tag).
function _wrapped_forwarddiff_ad()
    tag = ForwardDiff.Tag(NonlinearSolveTag(), Float64)
    return AutoForwardDiff{1, typeof(tag)}(tag)
end

# Stamp AutoForwardDiff with NonlinearSolveTag so duals match FunctionWrapper signatures.
# When the function has been wrapped via AutoSpecialize, also force chunksize=1 to match
# the precompiled N=1 dual type in the wrappers.
function standardize_forwarddiff_tag(
        ad::AutoForwardDiff{CS, Nothing}, prob::AbstractNonlinearProblem
    ) where {CS}
    prob.u0 isa Vector{Float64} || return ad
    if is_fw_wrapped(prob.f.f)
        return _wrapped_forwarddiff_ad()
    end
    tag = ForwardDiff.Tag(NonlinearSolveTag(), Float64)
    return AutoForwardDiff{CS, typeof(tag)}(tag)
end

# AutoPolyesterForwardDiff doesn't support custom tags. When the function is wrapped,
# replace it with AutoForwardDiff (chunksize=1, NonlinearSolveTag) so duals match wrappers.
function standardize_forwarddiff_tag(
        ad::AutoPolyesterForwardDiff, prob::AbstractNonlinearProblem
    )
    prob.u0 isa Vector{Float64} || return ad
    if is_fw_wrapped(prob.f.f)
        return _wrapped_forwarddiff_ad()
    end
    return ad
end

# Supported argument types for the norecompile pathway.
# NonlinearSolve IIP signature: f(du, u, p) -> Nothing

const NORECOMPILE_IIP_SUPPORTED_ARGS = Union{
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}},
    Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters},
    Tuple{Vector{dualT}, Vector{dualT}, Vector{Float64}},
    Tuple{Vector{dualT}, Vector{dualT}, SciMLBase.NullParameters},
    Tuple{Vector{dualT}, Vector{dualT}, Vector{dualT}},
    Tuple{Vector{dualT}, Vector{Float64}, Vector{dualT}},
}

# In-place argument and return type lists
const iip_arglists_default = (
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}},
    Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters},
    Tuple{Vector{dualT}, Vector{dualT}, Vector{Float64}},
    Tuple{Vector{dualT}, Vector{dualT}, SciMLBase.NullParameters},
    Tuple{Vector{dualT}, Vector{dualT}, Vector{dualT}},
    Tuple{Vector{dualT}, Vector{Float64}, Vector{dualT}},
)

const iip_returnlists_default = (
    Nothing, Nothing, Nothing, Nothing, Nothing, Nothing,
)

# IIP wrapfun: wraps f(du, u, p) with dual-aware type combinations.
# For Vector{Float64} state and Vector{Float64} params, uses the precomputed default lists.
@inline function wrapfun_iip(
        ff, inputs::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    )
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(ff), iip_arglists_default, iip_returnlists_default
    )
end

@inline function wrapfun_iip(
        ff, inputs::Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters}
    )
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(ff), iip_arglists_default, iip_returnlists_default
    )
end

# Generic typed method: generates dual arglists from actual input element types.
@inline function wrapfun_iip(
        ff, inputs::Tuple{T1, T2, T3}
    ) where {T1 <: AbstractVector, T2 <: AbstractVector, T3}
    T = eltype(T1)
    dT = dualgen(T)
    VdT = Vector{dT}
    iip_arglists = (
        Tuple{T1, T2, T3},
        Tuple{VdT, VdT, T3},
        Tuple{VdT, VdT, VdT},
        Tuple{VdT, T2, VdT},
    )
    iip_returnlists = (Nothing, Nothing, Nothing, Nothing)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(ff), iip_arglists, iip_returnlists
    )
end

const GENERAL_SOLVER_TYPES = [
    Nothing, NonlinearSolvePolyAlgorithm,
]

const DualNonlinearProblem = NonlinearProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}
const DualNonlinearLeastSquaresProblem = NonlinearLeastSquaresProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}
const DualAbstractNonlinearProblem = Union{
    DualNonlinearProblem, DualNonlinearLeastSquaresProblem,
}

function NonlinearSolveBase.additional_incompatible_backend_check(
        prob::AbstractNonlinearProblem, ::Union{AutoForwardDiff, AutoPolyesterForwardDiff}
    )
    return !ForwardDiff.can_dual(eltype(prob.u0))
end

Utils.value(::Type{Dual{T, V, N}}) where {T, V, N} = V
Utils.value(x::Dual) = ForwardDiff.value(x)
Utils.value(x::AbstractArray{<:Dual}) = Utils.value.(x)

function NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob::Union{
            IntervalNonlinearProblem, NonlinearProblem,
            ImmutableNonlinearProblem, NonlinearLeastSquaresProblem,
        },
        alg, args...; kwargs...
    )
    p = Utils.value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = Utils.value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        newprob = remake(prob; p, u0 = Utils.value(prob.u0))
    end

    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    fn = prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(prob, sol, uu) : prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, fn, uu, p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, fn, uu, p)
    z = -Jᵤ \ Jₚ
    pp = prob.p
    sumfun = ((z, p),) -> map(Base.Fix2(*, ForwardDiff.partials(p)), z)

    if uu isa Number
        partials = sum(sumfun, zip(z, pp))
    elseif p isa Number
        partials = sumfun((z, pp))
    else
        partials = sum(sumfun, zip(eachcol(z), pp))
    end

    return sol, partials
end

# Check if a NonlinearFunction has an AutoSpecialize-wrapped callable.
# Wrapping is only active for Vector{Float64} state/params, so the Number code paths
# (derivative, gradient) never encounter it — only the jacobian paths need configs.
_is_wrapped_nlf(f) = false
function _is_wrapped_nlf(f::SciMLBase.NonlinearFunction)
    return is_fw_wrapped(f.f)
end

const _nls_tag = ForwardDiff.Tag(NonlinearSolveTag(), Float64)

function NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        f2 = @closure p -> begin
            du = Utils.safe_similar(u, promote_type(eltype(u), eltype(p)))
            f(du, u, p)
            return du
        end
    else
        f2 = Base.Fix1(f, u)
    end
    if p isa Number
        return Utils.safe_reshape(ForwardDiff.derivative(f2, p), :, 1)
    elseif u isa Number
        return Utils.safe_reshape(ForwardDiff.gradient(f2, p), 1, :)
    else
        if _is_wrapped_nlf(f)
            cfg = ForwardDiff.JacobianConfig(f2, p, ForwardDiff.Chunk{1}(), _nls_tag)
            return ForwardDiff.jacobian(f2, p, cfg)
        end
        return ForwardDiff.jacobian(f2, p)
    end
end

function NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        jac_f = @closure((du, u) -> f(du, u, p))
        du_cache = Utils.safe_similar(u)
        if _is_wrapped_nlf(f)
            cfg = ForwardDiff.JacobianConfig(
                jac_f, du_cache, u, ForwardDiff.Chunk{1}(), _nls_tag
            )
            return ForwardDiff.jacobian(jac_f, du_cache, u, cfg)
        end
        return ForwardDiff.jacobian(jac_f, du_cache, u)
    end
    u isa Number && return ForwardDiff.derivative(Base.Fix2(f, p), u)
    if _is_wrapped_nlf(f)
        cfg = ForwardDiff.JacobianConfig(
            Base.Fix2(f, p), u, ForwardDiff.Chunk{1}(), _nls_tag
        )
        return ForwardDiff.jacobian(Base.Fix2(f, p), u, cfg)
    end
    return ForwardDiff.jacobian(Base.Fix2(f, p), u)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(
        u::Number, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}
    ) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(
        u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}
    ) where {T, V, P}
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, Utils.restructure(u, partials)))
end

for algType in GENERAL_SOLVER_TYPES
    @eval function SciMLBase.__solve(
            prob::DualAbstractNonlinearProblem, alg::$(algType), args...; kwargs...
        )
        sol,
            partials = NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
            prob, alg, args...; kwargs...
        )
        dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
        )
    end
end

function InternalAPI.reinit!(
        cache::NonlinearSolveForwardDiffCache, args...;
        p = cache.p, u0 = NonlinearSolveBase.get_u(cache.cache), kwargs...
    )
    InternalAPI.reinit!(
        cache.cache; p = NonlinearSolveBase.nodual_value(p),
        u0 = NonlinearSolveBase.nodual_value(u0), kwargs...
    )
    cache.p = p
    cache.values_p = NonlinearSolveBase.nodual_value(p)
    cache.partials_p = ForwardDiff.partials(p)
    return cache
end

for algType in GENERAL_SOLVER_TYPES
    @eval function SciMLBase.__init(
            prob::DualAbstractNonlinearProblem, alg::$(algType), args...; kwargs...
        )
        p = NonlinearSolveBase.nodual_value(prob.p)
        newprob = SciMLBase.remake(prob; u0 = NonlinearSolveBase.nodual_value(prob.u0), p)
        cache = init(newprob, alg, args...; kwargs...)
        return NonlinearSolveForwardDiffCache(
            cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p)
        )
    end
end

function CommonSolve.solve!(cache::NonlinearSolveForwardDiffCache)
    sol = solve!(cache.cache)
    prob = cache.prob
    uu = sol.u

    fn = prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(prob, sol, uu) : prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, fn, uu, cache.values_p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, fn, uu, cache.values_p)

    z_arr = -Jᵤ \ Jₚ

    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if cache.p isa Number
        partials = sumfun((z_arr, cache.p))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), cache.p))
    end

    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, cache.p)
    return SciMLBase.build_solution(
        prob, cache.alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

NonlinearSolveBase.nodual_value(x) = x
NonlinearSolveBase.nodual_value(x::Dual) = ForwardDiff.value(x)
NonlinearSolveBase.nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

@inline NonlinearSolveBase.pickchunksize(x) = pickchunksize(length(x))
@inline NonlinearSolveBase.pickchunksize(x::Int) = ForwardDiff.pickchunksize(x)

end
