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
    NonlinearProblem, NonlinearLeastSquaresProblem, ImmutableNonlinearProblem, remake
using SciMLStructures: SciMLStructures
using Setfield: @set

using LinearAlgebra: LinearAlgebra, dot, norm
using NonlinearSolveBase: NonlinearSolveBase, Utils, InternalAPI,
    NonlinearSolvePolyAlgorithm, NonlinearSolveForwardDiffCache,
    NonlinearSolveTag, is_fw_wrapped

import NonlinearSolveBase: wrapfun_iip, standardize_forwarddiff_tag

const DI = DifferentiationInterface

_cache_storage_type(
    ::FunctionWrappersWrappers.FunctionWrappersWrapper{FW, P, CS}
) where {FW, P, CS} = CS

# Derive the concrete storage type through the public cache-mode constructor.
const FWW_SINGLE_CACHE_STORAGE_TYPE = _cache_storage_type(
    FunctionWrappersWrappers.FunctionWrappersWrapper(
        identity, (Tuple{Float64},), (Float64,);
        cache = FunctionWrappersWrappers.SingleCache(),
        policy = FunctionWrappersWrappers.AllowNonIsBits(),
    )
)

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

# Construct the `FunctionWrappersWrapper` bypassing the convenience constructor, mirroring
# DiffEqBase's `_make_fww` (ext/DiffEqBaseForwardDiffExt.jl). The convenience constructor
# builds its wrapper tuple with `map(argtypes, rettypes) do A, R; …; end`, whose closure
# leaves `A`/`R` inferred as the join of the heterogeneous `argtypes` tuple, so the wrapper
# type — and every cache built from the wrapped problem — widens to abstract whenever the
# wrapped callable is one inference cannot see through: a functor such as the homotopy
# drivers' `FixLambda` / `AugmentedHomotopy` / `HomotopyResidual` residuals (the same reason
# DiffEqBase hit this with many-parameter `ODEFunction`s). Binding each arglist as a
# `::Type{A}` type parameter makes `FunctionWrapper{Nothing, A}(vff)` fully inferable, so
# `typeof(fwt)` — and the wrapper — stays concrete. `vff` is `@nospecialize`d because it is
# already type-erased through `Void` (that is what makes the norecompile path precompile).
function _make_fww_iip(
        @nospecialize(vff), ::Type{A1}, ::Type{A2}, ::Type{A3}, ::Type{A4}
    ) where {A1, A2, A3, A4}
    FW = FunctionWrappers.FunctionWrapper
    FWT = Tuple{
        FW{Nothing, A1}, FW{Nothing, A2},
        FW{Nothing, A3}, FW{Nothing, A4},
    }
    return FunctionWrappersWrappers.FunctionWrappersWrapper{
        FWT, FunctionWrappersWrappers.AllowNonIsBits, FWW_SINGLE_CACHE_STORAGE_TYPE,
    }(vff)
end

# IIP wrapfun: wraps f(du, u, p) with dual-aware type combinations.
# Works for any `AbstractArray` state; the Dual-eltype array type `VdT` is
# derived via `typeof(similar(u0, dT))` so signatures follow the user's
# concrete array kind (plain `Vector{Float64}` → `Vector{Dual}`,
# `Array{Float64, 3}` → `Array{Dual, 3}`, `CuArray{Float32}` →
# `CuArray{Dual}`, etc.). The call allocates once at FWW-construction time
# (not on any hot path) and is broadly compatible with array kinds that do
# not implement `ArrayInterface.promote_eltype` — e.g. `CuArray` — which was
# breaking GPU tests when this derived `VdT` via `promote_eltype`.
@inline function wrapfun_iip(
        ff, inputs::Tuple{T1, T2, T3}
    ) where {T1 <: AbstractArray, T2 <: AbstractArray, T3}
    T = eltype(T1)
    dT = dualgen(T)
    VdT = typeof(similar(inputs[1], dT))
    return _make_fww_iip(
        SciMLBase.Void(ff),
        Tuple{T1, T2, T3},
        Tuple{VdT, VdT, T3},
        Tuple{VdT, VdT, VdT},
        Tuple{VdT, T2, VdT},
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

# --------------------------------------------------------------------------
# Structured-parameter Dual handling
#
# `DualAbstractNonlinearProblem` can only see problems whose `p` field is literally
# a `Dual` or an `AbstractArray{<:Dual}`. Duals also arrive nested inside `p` —
# a `NamedTuple`, `Tuple`, non-concretely-typed array, or a `SciMLStructures`
# parameter container (`MTKParameters`, `DespecializedParameters`, …) — or inside
# `u0`/`tspan` alone. Those problems miss the type-based dispatch above and would
# run the solver's internal state in Dual arithmetic, where e.g. quasi-Newton
# updates divide by denominators that vanish in value but not in partials, while
# termination checks inspect values only.
#
# `_dual_view(x)` flattens every `Dual` leaf of `x` into
# `(; values, rebuild, duals)`: `values` is the flat primal vector,
# `rebuild(θ)` splices a flat vector (of any number type) back into `x`'s
# structure, and `duals` is the flat vector of original `Dual` leaves used for
# partial extraction. Returns `nothing` when `x` contains no Duals.

_dual_view(::Nothing) = nothing
_dual_view(::Number) = nothing
_dual_view(x::Dual) = (; values = [ForwardDiff.value(x)], rebuild = first, duals = [x])

function _dual_view(x::AbstractArray{<:Dual})
    return (;
        values = vec(map(ForwardDiff.value, x)),
        rebuild = θ -> Utils.restructure(x, θ),
        duals = vec(x),
    )
end

# Combine per-item views of a container into a view of the whole. `assemble` maps a
# vector of rebuilt items back to the container type.
function _join_dual_views(items::Vector, subs::Vector, assemble)
    all(isnothing, subs) && return nothing
    ranges = Vector{UnitRange{Int}}(undef, length(subs))
    dual_parts = Any[]
    off = 0
    for i in eachindex(subs)
        s = subs[i]
        if s === nothing
            ranges[i] = 1:0
        else
            ranges[i] = (off + 1):(off + length(s.values))
            off = last(ranges[i])
            push!(dual_parts, s.duals)
        end
    end
    isempty(dual_parts) && return nothing
    rebuild = let subs = subs, items = items, ranges = ranges, assemble = assemble
        θ -> assemble(
            Any[
                subs[i] === nothing ? items[i] : subs[i].rebuild(θ[ranges[i]])
                    for i in eachindex(subs)
            ]
        )
    end
    return (; values = _concat_values(dual_parts), rebuild, duals = _concat_duals(dual_parts))
end

function _dual_view(x::AbstractArray)
    isconcretetype(eltype(x)) && return nothing
    subs = Any[_dual_view(v) for v in x]
    return _join_dual_views(collect(x), subs, v -> Utils.restructure(x, v))
end

function _dual_view(x::Tuple)
    subs = Any[_dual_view(v) for v in x]
    return _join_dual_views(collect(x), subs, Tuple)
end

function _dual_view(x::NamedTuple)
    subs = Any[_dual_view(v) for v in values(x)]
    return _join_dual_views(
        collect(values(x)), subs, v -> NamedTuple{keys(x)}(Tuple(v))
    )
end

function _dual_view(x::AbstractDict)
    items = collect(values(x))
    subs = Any[_dual_view(v) for v in items]
    all(isnothing, subs) && return nothing
    ks = collect(keys(x))
    # `convert(::Type{<:Dual}, ::Real)` exists, so a same-type probe like
    # `Dict{Symbol, Dual}(:a => 3.0)` succeeds while re-wrapping the primal as a
    # Dual — only keep the same-type constructor when the probed container is
    # actually Dual-free.
    vals = Any[
        subs[i] === nothing ? items[i] : subs[i].rebuild(subs[i].values)
            for i in eachindex(subs)
    ]
    probe = try
        typeof(x)(zip(ks, vals))
    catch
        nothing
    end
    assemble = if probe !== nothing && SciMLBase.anyeltypedual(probe) === Any
        v -> typeof(x)(zip(ks, v))
    else
        v -> Dict(zip(ks, v))
    end
    return _join_dual_views(items, subs, assemble)
end

# SciMLStructures portions that may carry solver-relevant Dual values. `Initials`
# only exists in newer SciMLStructures versions.
const _FD_PORTIONS = Tuple(
    Iterators.filter(
        !isnothing,
        Any[
            SciMLStructures.Tunable(),
            isdefined(SciMLStructures, :Initials) ? SciMLStructures.Initials() : nothing,
            SciMLStructures.Discrete(),
            SciMLStructures.Constants(),
            SciMLStructures.Caches(),
            SciMLStructures.Input(),
        ],
    )
)

function _dual_view(x)
    SciMLStructures.isscimlstructure(x) || return nothing
    replacers = Function[]
    ranges = UnitRange{Int}[]
    dual_parts = Any[]
    off = 0
    for S in _FD_PORTIONS
        # `hasportion` is not reliably implemented (e.g. `MTKParameters` omits it
        # and `DespecializedParameters` forwards it generically), so probe
        # `canonicalize` directly: a `MethodError` — including one raised inside a
        # forwarding wrapper — means the portion is unsupported.
        buf = try
            SciMLStructures.canonicalize(S, x)[1]
        catch err
            err isa MethodError || rethrow()
            continue
        end
        buf === nothing && continue
        if buf isa Dual || (buf isa AbstractArray && eltype(buf) <: Dual)
            flat = buf isa Dual ? [buf] : vec(buf)
            n = length(flat)
            push!(ranges, (off + 1):(off + n))
            push!(
                replacers,
                @closure((q, slice) -> SciMLStructures.replace(S, q, collect(slice)))
            )
            push!(dual_parts, flat)
            off += n
        elseif buf isa AbstractArray && !isconcretetype(eltype(buf))
            sub = _dual_view(vec(collect(buf)))
            sub === nothing && continue
            n = length(sub.values)
            push!(ranges, (off + 1):(off + n))
            push!(
                replacers,
                @closure(
                    (q, slice) -> SciMLStructures.replace(S, q, sub.rebuild(slice))
                )
            )
            push!(dual_parts, sub.duals)
            off += n
        end
    end
    isempty(dual_parts) && return nothing
    rebuild = @closure θ -> begin
        q = x
        for i in eachindex(replacers)
            q = replacers[i](q, θ[ranges[i]])
        end
        return q
    end
    return (; values = _concat_values(dual_parts), rebuild, duals = _concat_duals(dual_parts))
end

function _concat_duals(parts)
    isempty(parts) && return nothing
    D = mapreduce(eltype, promote_type, parts)
    out = Vector{D}(undef, sum(length, parts))
    i = 1
    for part in parts
        for d in part
            out[i] = d
            i += 1
        end
    end
    return out
end

_concat_values(parts) = ForwardDiff.value.(_concat_duals(parts))

# Strip Dual leaves wherever they appear; non-dual inputs pass through unchanged.
_strip_duals(x) = (v = _dual_view(x)) === nothing ? x : v.rebuild(v.values)

function NonlinearSolveBase.nodual_value(x)
    return _strip_duals(x)
end

# Cheap type-level gate mirroring SciMLBase's `promote_u0` detection so that the
# flattening machinery only runs when duals are actually present.
function _problem_has_duals(prob)
    hasproperty(prob, :p) && SciMLBase.anyeltypedual(prob.p) !== Any && return true
    hasproperty(prob, :u0) && SciMLBase.anyeltypedual(prob.u0) !== Any && return true
    hasproperty(prob, :tspan) && SciMLBase.anyeltypedual(prob.tspan) !== Any &&
        return true
    return false
end

# If the flat primals are Dual-free but the rebuilt object still carries Duals,
# the container's repack re-wraps values as Duals (e.g. a concrete Dual-typed
# buffer written via `setindex!`-style `replace`) — no strip progress, so
# recursing would loop forever. Exempt nested Duals (`values` still Dual): those
# legitimately strip one level per recursive dispatch.
function _check_stripped(view, stripped, name)
    view === nothing && return
    SciMLBase.anyeltypedual(view.values) === Any || return
    SciMLBase.anyeltypedual(stripped) === Any && return
    throw(
        ArgumentError(
            "failed to strip ForwardDiff Duals from problem $name; the " *
                "$(typeof(stripped)) container re-wraps primal values as Duals",
        ),
    )
end

function _forwarddiff_rebuild_prob(prob)
    pview = _dual_view(prob.p)
    p = pview === nothing ? prob.p : pview.rebuild(pview.values)
    _check_stripped(pview, p, "parameters")
    newprob = if prob isa IntervalNonlinearProblem
        tspan = map(_strip_duals, prob.tspan)
        IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        u0view = _dual_view(prob.u0)
        u0 = u0view === nothing ? prob.u0 : u0view.rebuild(u0view.values)
        _check_stripped(u0view, u0, "u0")
        remake(prob; p, u0)
    end
    # `remake` reuses `prob.f.f`. If `get_concrete_problem` had wrapped the outer
    # prob under a Dual u0 eltype (via `promote_u0`), the stored
    # `FunctionWrappersWrapper` signatures are keyed off that Dual eltype and would
    # miss the inner value-typed solve's `f(du, u, p)` dispatch. Unwrap here so the
    # inner solve's `maybe_wrap_f` rebuilds a wrapper aligned with the value-typed
    # `u0`/`p`.
    if is_fw_wrapped(newprob.f.f)
        newprob = @set newprob.f.f = NonlinearSolveBase.get_raw_f(newprob.f.f)
    end
    return newprob, pview
end

function _forwarddiff_∂f_∂p(prob, f::F, u, pview) where {F}
    rebuild = pview.rebuild
    if SciMLBase.isinplace(prob)
        f2 = @closure θ -> begin
            du = Utils.safe_similar(u, promote_type(eltype(u), eltype(θ)))
            f(du, u, rebuild(θ))
            return du
        end
    else
        f2 = @closure θ -> f(u, rebuild(θ))
    end
    if u isa Number
        return Utils.safe_reshape(ForwardDiff.gradient(f2, pview.values), 1, :)
    else
        return ForwardDiff.jacobian(f2, pview.values)
    end
end

function _forwarddiff_combine_partials(z, duals, uu)
    pvec = ForwardDiff.partials.(duals)
    uu isa Number && return sum(zᵢ * pᵢ for (zᵢ, pᵢ) in zip(vec(z), pvec))
    return z * pvec
end

# A converged root does not depend on the initial guess: Duals in `u0`/`tspan`
# contribute structurally zero partials, attached under the Dual tag they carried.
function _zero_partials(uu, duals)
    z = zero(ForwardDiff.partials(duals[1]))
    uu isa Number && return z
    return fill(z, length(uu))
end

function _forwarddiff_solve_impl(prob, args...; kwargs...)
    newprob, pview = _forwarddiff_rebuild_prob(prob)
    # `pview === nothing` while `prob.p` still carries Duals means the parameter
    # structure cannot be rebuilt — error loudly rather than run the solve in
    # Dual arithmetic. Duals remaining after a successful strip (nested Duals)
    # are handled by recursive dispatch into the inner solve.
    if pview === nothing && hasproperty(prob, :p) &&
            SciMLBase.anyeltypedual(prob.p) !== Any
        throw(
            ArgumentError(
                "failed to strip ForwardDiff Duals from problem parameters; the " *
                    "parameter object $(typeof(prob.p)) contains Duals in a structure " *
                    "NonlinearSolve cannot rebuild",
            ),
        )
    end

    sol = solve(newprob, args...; kwargs...)
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

    if pview === nothing
        # Only `u0`/`tspan` carried Duals: the root's sensitivity to the guess is
        # structurally zero; attach zero partials under the incoming tag.
        duals = if hasproperty(prob, :u0)
            _dual_view(prob.u0)
        else
            nothing
        end
        duals === nothing && hasproperty(prob, :tspan) &&
            (duals = _dual_view(prob.tspan))
        duals === nothing && return nothing
        partials = _zero_partials(uu, duals.duals)
        dualsrc = duals.duals
    else
        p = pview.rebuild(pview.values)
        Jₚ = _forwarddiff_∂f_∂p(ad_prob, fn, uu, pview)
        Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(ad_prob, fn, uu, p)
        z = -_forwarddiff_implicit_solve(Jᵤ, Jₚ)
        partials = _forwarddiff_combine_partials(z, pview.duals, uu)
        dualsrc = pview.duals
    end

    return sol, partials, dualsrc
end

function NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob::Union{
            IntervalNonlinearProblem, NonlinearProblem,
            ImmutableNonlinearProblem, NonlinearLeastSquaresProblem,
        },
        alg, args...; kwargs...
    )
    sol, partials, _ = _forwarddiff_solve_impl(prob, alg, args...; kwargs...)
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

_forwarddiff_implicit_solve(A, B) = NonlinearSolveBase.implicit_sensitivity_solve(A, B)

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

# Structured-parameter flattening may yield an abstractly-typed leaf vector (e.g.
# `Vector{Any}`); recover the Dual tag from the first leaf.
function NonlinearSolveBase.nonlinearsolve_dual_solution(
        u, partials, p::AbstractArray
    )
    return nonlinearsolve_dual_solution(u, partials, p[firstindex(p)])
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
    pview = _dual_view(p)
    u0view = _dual_view(u0)
    stripped_p = pview === nothing ? p : pview.rebuild(pview.values)
    stripped_u0 = u0view === nothing ? u0 : u0view.rebuild(u0view.values)
    _check_stripped(pview, stripped_p, "parameters")
    _check_stripped(u0view, stripped_u0, "u0")
    InternalAPI.reinit!(
        cache.cache; p = stripped_p, u0 = stripped_u0, kwargs...
    )
    cache.p = p
    cache.values_p = stripped_p
    partials_p = pview === nothing ? nothing : ForwardDiff.partials.(pview.duals)
    # `partials_p` is concretely typed by the constructor under `@concrete`;
    # it is never read, so keep the stale value rather than throw when a
    # reinit changes whether `p` carries duals.
    partials_p isa typeof(cache.partials_p) && (cache.partials_p = partials_p)
    cache.u0duals = u0view === nothing ? nothing : u0view.duals
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

    pview = _dual_view(cache.p)
    if pview === nothing
        # Duals only in `u0`: sensitivity to the guess is structurally zero.
        # `reinit!` with fully primal inputs leaves no Dual source at all —
        # the primal solution is then the answer.
        cache.u0duals === nothing && return sol
        dualsrc = cache.u0duals
        partials = _zero_partials(uu, dualsrc)
    else
        Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(ad_prob, fn, uu, cache.values_p)
        Jₚ = _forwarddiff_∂f_∂p(ad_prob, fn, uu, pview)
        z_arr = -_forwarddiff_implicit_solve(Jᵤ, Jₚ)
        partials = _forwarddiff_combine_partials(z_arr, pview.duals, uu)
        dualsrc = pview.duals
    end

    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, dualsrc)
    return SciMLBase.build_solution(
        prob, cache.alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

# Interception hooks called from `solve_call`/`init_call` for problems the
# `DualAbstractNonlinearProblem` dispatch cannot match — Duals nested inside a
# structured `p`, or Duals only in `u0`/`tspan`. The nominal dispatch stays the
# fast path; these run first and return `nothing` for non-Dual problems.
const _FD_HOOK_PROBLEMS = Union{
    IntervalNonlinearProblem, NonlinearProblem,
    ImmutableNonlinearProblem, NonlinearLeastSquaresProblem,
}

function InternalAPI.forwarddiff_solve(prob::AbstractNonlinearProblem, args...; kwargs...)
    prob isa DualAbstractNonlinearProblem && return nothing
    prob isa _FD_HOOK_PROBLEMS || return nothing
    _problem_has_duals(prob) || return nothing
    res = _forwarddiff_solve_impl(prob, args...; kwargs...)
    res === nothing && return nothing
    sol, partials, dualsrc = res
    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, dualsrc)
    alg = length(args) > 0 ? args[1] :
        (hasproperty(sol, :alg) ? sol.alg : nothing)
    # Bracketing solutions carry `left`/`right` endpoints — both approximate the
    # root within tolerance, so they take the same implicit partials.
    extra = if hasproperty(sol, :left) && sol.left !== nothing
        (;
            left = NonlinearSolveBase.nonlinearsolve_dual_solution(
                sol.left, partials, dualsrc
            ),
            right = NonlinearSolveBase.nonlinearsolve_dual_solution(
                sol.right, partials, dualsrc
            ),
        )
    else
        (;)
    end
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original, extra...
    )
end

function InternalAPI.forwarddiff_init(prob::AbstractNonlinearProblem, args...; kwargs...)
    prob isa DualAbstractNonlinearProblem && return nothing
    # `init` requires a `u0`-carrying problem; IntervalNonlinearProblem only
    # supports `solve`.
    prob isa Union{
        NonlinearProblem, ImmutableNonlinearProblem, NonlinearLeastSquaresProblem,
    } || return nothing
    _problem_has_duals(prob) || return nothing
    newprob, pview = _forwarddiff_rebuild_prob(prob)
    if pview === nothing && hasproperty(prob, :p) &&
            SciMLBase.anyeltypedual(prob.p) !== Any
        throw(
            ArgumentError(
                "failed to strip ForwardDiff Duals from problem parameters; the " *
                    "parameter object $(typeof(prob.p)) contains Duals in a structure " *
                    "NonlinearSolve cannot rebuild",
            ),
        )
    end
    cache = init(newprob, args...; kwargs...)
    alg = length(args) > 0 ? args[1] :
        (hasproperty(cache, :alg) ? cache.alg : nothing)
    if pview === nothing
        dualsrc = hasproperty(prob, :u0) ? _dual_view(prob.u0) : nothing
        dualsrc === nothing && hasproperty(prob, :tspan) &&
            (dualsrc = _dual_view(prob.tspan))
        dualsrc === nothing && return nothing
        u0duals = dualsrc.duals
        partials_p = nothing
    else
        u0duals = nothing
        partials_p = ForwardDiff.partials.(pview.duals)
    end
    return NonlinearSolveForwardDiffCache(
        cache, newprob, alg, prob.p, newprob.p, partials_p, u0duals
    )
end

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
