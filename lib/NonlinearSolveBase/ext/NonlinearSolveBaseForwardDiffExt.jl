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
using Setfield: @set

using LinearAlgebra: LinearAlgebra, dot, norm
using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem, Utils, InternalAPI,
    NonlinearSolvePolyAlgorithm, NonlinearSolveForwardDiffCache,
    NonlinearSolveTag, is_fw_wrapped

import NonlinearSolveBase: wrapfun_iip, standardize_forwarddiff_tag

const DI = DifferentiationInterface

# --- AutoSpecialize / norecompile infrastructure for ForwardDiff ---

const dualT = ForwardDiff.Dual{
    ForwardDiff.Tag{NonlinearSolveTag, Float64}, Float64, 1,
}
dualgen(::Type{T}) where {T} = ForwardDiff.Dual{
    ForwardDiff.Tag{NonlinearSolveTag, T}, T, 1,
}

# Helper: build the canonical AutoForwardDiff for wrapped functions
# (chunksize=1 + NonlinearSolveTag). The tag's `V` parameter takes the
# actual problem eltype rather than being hardcoded to `Float64`, so the
# stamped AD backend properly reflects the user's problem type.
function _wrapped_forwarddiff_ad(::Type{T}) where {T}
    tag = ForwardDiff.Tag(NonlinearSolveTag(), T)
    return AutoForwardDiff{1, typeof(tag)}(tag)
end

# Stamp AutoForwardDiff with NonlinearSolveTag so duals match the wrapped
# `FunctionWrappersWrapper` signatures. Only stamps when the user function was
# actually wrapped via AutoSpecialize — otherwise leaves `ad` untouched so
# DifferentiationInterface generates a fresh runtime tag from the function type.
# Substituting the canonical tag in the non-wrapped path would otherwise drag in
# a precompile-time `@generated tagcount` literal that can `≺`-reverse against
# tags created later for nested ForwardDiff over an inner solve.
function standardize_forwarddiff_tag(
        ad::AutoForwardDiff{CS, Nothing}, prob::AbstractNonlinearProblem
    ) where {CS}
    is_fw_wrapped(prob.f.f) || return ad
    return _wrapped_forwarddiff_ad(eltype(prob.u0))
end

# AutoPolyesterForwardDiff doesn't support custom tags. When the function is
# wrapped, replace it with AutoForwardDiff (chunksize=1, NonlinearSolveTag) so
# duals match wrappers. Otherwise leave it alone.
function standardize_forwarddiff_tag(
        ad::AutoPolyesterForwardDiff, prob::AbstractNonlinearProblem
    )
    is_fw_wrapped(prob.f.f) || return ad
    return _wrapped_forwarddiff_ad(eltype(prob.u0))
end

# IIP wrapfun: wraps f(du, u, p) with dual-aware type combinations.
# Works for any `AbstractArray` state; `ArrayInterface.promote_eltype` produces
# the Dual-eltype array type at the type level (no allocation) so signatures
# follow the user's concrete array kind (plain `Vector{Float64}` →
# `Vector{Dual}`, `Array{Float64, 3}` → `Array{Dual, 3}`, etc.).
@inline function wrapfun_iip(
        ff, inputs::Tuple{T1, T2, T3}
    ) where {T1 <: AbstractArray, T2 <: AbstractArray, T3}
    T = eltype(T1)
    dT = dualgen(T)
    VdT = ArrayInterface.promote_eltype(T1, dT)
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
        # `remake` reuses `prob.f.f`. If `get_concrete_problem` had wrapped the
        # outer prob under a Dual u0 eltype (via `promote_u0`), the stored
        # `FunctionWrappersWrapper` signatures are keyed off that Dual eltype
        # and would miss the inner Float64 solve's `f(du, u, p)` dispatch.
        # Unwrap here so the inner solve's `maybe_wrap_f` rebuilds a wrapper
        # aligned with the value-typed `u0`/`p`.
        if is_fw_wrapped(newprob.f.f)
            newprob = @set newprob.f.f = NonlinearSolveBase.get_raw_f(newprob.f.f)
        end
    end

    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    # Unwrap AutoSpecializeCallable for the AD-over-solve Jacobian computations.
    # These use ForwardDiff with closure-based tags that don't match the wrapper signatures.
    ad_prob = if is_fw_wrapped(prob.f.f)
        @set prob.f.f = NonlinearSolveBase.get_raw_f(prob.f.f)
    else
        prob
    end

    fn = ad_prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(ad_prob, sol, uu) : ad_prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(ad_prob, fn, uu, p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(ad_prob, fn, uu, p)
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
        return ForwardDiff.jacobian(f2, p)
    end
end

function NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        jac_f = @closure((du, u) -> f(du, u, p))
        du_cache = Utils.safe_similar(u)
        return ForwardDiff.jacobian(jac_f, du_cache, u)
    end
    u isa Number && return ForwardDiff.derivative(Base.Fix2(f, p), u)
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
        # See comment in `nonlinearsolve_forwarddiff_solve`: the outer FWW's
        # signatures were built under a Dual u0 eltype and would miss the
        # inner value-typed solve. Unwrap and let the inner init rebuild.
        if is_fw_wrapped(newprob.f.f)
            newprob = @set newprob.f.f = NonlinearSolveBase.get_raw_f(newprob.f.f)
        end
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

    # Unwrap AutoSpecializeCallable for the AD-over-solve Jacobian computations.
    ad_prob = if is_fw_wrapped(prob.f.f)
        @set prob.f.f = NonlinearSolveBase.get_raw_f(prob.f.f)
    else
        prob
    end

    fn = ad_prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(ad_prob, sol, uu) : ad_prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(ad_prob, fn, uu, cache.values_p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(ad_prob, fn, uu, cache.values_p)

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

# Precompile common Dual number operations to reduce first-solve latency.
# Nonlinear solvers compute Jacobians via ForwardDiff, triggering compilation of
# Dual arithmetic, broadcast, and SubArray patterns at runtime. Exercising these
# patterns here moves that overhead to precompile time.
# NonlinearSolveTag and dualT are already defined at the top of this extension.

import PrecompileTools
PrecompileTools.@compile_workload begin
    # Scalar operations on Dual numbers (arithmetic, math functions, comparisons)
    d1 = dualT(1.0, ForwardDiff.Partials((0.5,)))
    d2 = dualT(2.0, ForwardDiff.Partials((1.0,)))
    s = 3.14

    # Arithmetic: Dual-Dual and Dual-scalar
    d1 + d2
    d1 - d2
    d1 * d2
    d1 / d2
    d1 + s
    s + d1
    d1 - s
    s - d1
    d1 * s
    s * d1
    d1 / s
    s / d1
    -d1
    abs(d1)

    # Powers and roots
    d1^2
    d1^3
    d2^0.5
    sqrt(d2)
    cbrt(d2)

    # Transcendental functions
    exp(d1)
    log(d2)
    sin(d1)
    cos(d1)
    tan(d1)
    asin(dualT(0.5, ForwardDiff.Partials((1.0,))))
    acos(dualT(0.5, ForwardDiff.Partials((1.0,))))
    atan(d1)
    atan(d1, d2)
    sinh(d1)
    cosh(d1)
    tanh(d1)

    # Comparisons and predicates
    d1 < d2
    d1 > d2
    d1 <= d2
    d1 >= d2
    d1 == d2
    isnan(d1)
    isinf(d1)
    isfinite(d1)

    # min/max (used in convergence checks, damping)
    min(d1, d2)
    max(d1, d2)
    min(d1, s)
    max(d1, s)

    # Conversion and promotion
    zero(dualT)
    one(dualT)
    float(d1)
    ForwardDiff.value(d1)
    ForwardDiff.partials(d1)

    # Array operations on Vector{dualT}
    v1 = [d1, d2, dualT(0.0, ForwardDiff.Partials((0.0,)))]
    v2 = [d2, d1, dualT(1.0, ForwardDiff.Partials((0.1,)))]

    # Basic array ops
    v1 + v2
    v1 - v2
    v1 .* v2
    v1 ./ v2
    s .* v1
    v1 .+ s
    v1 .- s
    v1 .^ 2
    v1 .^ 0.5

    # In-place array operations
    out = similar(v1)
    out .= v1 .+ v2
    out .= v1 .- v2
    out .= v1 .* v2
    out .= s .* v1
    out .= v1 .* s .+ v2
    out .= v1 .* s .- v2 .* s

    # Reductions (used in norm calculations, convergence checks)
    sum(v1)
    sum(abs2, v1)
    maximum(abs, v1)

    # LinearAlgebra operations
    dot(v1, v2)
    norm(v1)
    norm(v1, Inf)
    norm(v1, 1)

    # copy / fill
    copy(v1)
    fill!(out, zero(dualT))

    # SubArray broadcast operations for Float64 and Dual types.
    # Nonlinear functions that use @view with broadcast (e.g. residual computations
    # on subsets of state) trigger compilation of deeply-nested Broadcasted types
    # for SubArray at runtime. Exercising common patterns here moves that
    # compilation from first-solve to precompile time.
    for T in (Float64, dualT)
        x = zeros(T, 6)
        dx = zeros(T, 6)
        sv1 = @view x[1:2]
        sv2 = @view x[3:4]
        sv3 = @view x[5:6]
        dsv1 = @view dx[1:2]
        dsv2 = @view dx[3:4]
        dsv3 = @view dx[5:6]
        k = 0.04

        # Common broadcast patterns from nonlinear residual functions
        # Pattern 1a: dst .= -k .* src1 .+ k .* src2 .* src3
        dsv1 .= .-k .* sv1 .+ k .* sv2 .* sv3
        # Pattern 1b: dst .= k .* src1 .+ k .* src2 .* src3
        dsv1 .= k .* sv1 .+ k .* sv2 .* sv3
        # Pattern 2: dst .= k .* src1 .- k .* src2 .^ 2 .- k .* src2 .* src3
        dsv2 .= k .* sv1 .- k .* sv2 .^ 2 .- k .* sv2 .* sv3
        # Pattern 3: dst .= k .* src .^ 2
        dsv3 .= k .* sv2 .^ 2

        # Additional SubArray patterns
        # Simple assignment and scaling
        dsv1 .= sv1
        dsv1 .= k .* sv1
        dsv1 .= sv1 .+ sv2
        dsv1 .= sv1 .- sv2
        dsv1 .= sv1 .* sv2
        # Negation patterns
        dsv1 .= .-sv1
        dsv1 .= .-sv1 .+ sv2
    end
end

end
