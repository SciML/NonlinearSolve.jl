"""
    NonlinearSolveTag

Tag type used for ForwardDiff dual number tagging in the nonlinear solver norecompile
infrastructure. Analogous to `OrdinaryDiffEqTag` in DiffEqBase.
"""
struct NonlinearSolveTag end

const NORECOMPILE_ARGUMENT_MESSAGE = """
No-recompile mode (AutoSpecialize) for nonlinear solvers is only supported for state
arguments of type `Vector{Float64}`, and parameter arguments of type `Vector{Float64}`
or `SciMLBase.NullParameters`.
"""

struct NoRecompileArgumentError <: Exception
    args::Any
end

function Base.showerror(io::IO, e::NoRecompileArgumentError)
    println(io, NORECOMPILE_ARGUMENT_MESSAGE)
    print(io, "Attempted arguments: ")
    return print(io, e.args)
end

function unwrap_fw(fw::FunctionWrappers.FunctionWrapper)
    return fw.obj[]
end

"""
    AutoSpecializeCallable{FW}

Holds both the `FunctionWrappersWrapper` (for type-restricted dispatch through precompiled
wrappers) and the original callable function (type-erased as `Any` so all wrapped functions
share the same Julia type, enabling precompilation). The original is used as a fallback
for external packages that call with unsupported dual types (IIP path) and for nested
ForwardDiff differentiation in NLLS (via `get_raw_f`).
"""
struct AutoSpecializeCallable{FW}
    fw::FW
    orig::Any  # type-erased: all wrapped functions share the same Julia type
end

# Call through FunctionWrappersWrapper. All argument types that reach here have
# matching wrapper signatures (wrapping is only applied for supported types).
@inline (f::AutoSpecializeCallable)(args...) = f.fw(args...)

"""
    is_fw_wrapped(f) -> Bool

Return `true` if the function `f` has been wrapped by the AutoSpecialize infrastructure.
"""
is_fw_wrapped(f) = false
is_fw_wrapped(::AutoSpecializeCallable) = true

"""
    get_raw_f(f)

If `f` has been wrapped by AutoSpecialize, return the original unwrapped callable.
Otherwise return `f` unchanged.
"""
get_raw_f(f) = f
get_raw_f(f::AutoSpecializeCallable) = f.orig

"""
    _uses_enzyme_ad(ad) -> Bool

Return `true` if `ad` is an Enzyme-based AD backend (possibly wrapped in `AutoSparse`).
Enzyme cannot differentiate through FunctionWrappers' `llvmcall`, so
`AutoSpecializeCallable` must be unwrapped before passing to DI with Enzyme.
"""
_uses_enzyme_ad(::ADTypes.AutoEnzyme) = true
_uses_enzyme_ad(ad::AutoSparse) = _uses_enzyme_ad(ADTypes.dense_ad(ad))
_uses_enzyme_ad(_) = false

"""
    _uses_ad_sparsity_detector(ad) -> Bool

Return `true` if `ad` is an `AutoSparse` whose sparsity detector differentiates the
function (`DenseSparsityDetector`). Such detectors call the function with their own AD
tag, which `AutoSpecializeCallable`'s FunctionWrapper does not have an entry for, so the
raw user function must be used. Tracer-based detectors do not differentiate and are fine.
"""
_uses_ad_sparsity_detector(ad::AutoSparse) = ADTypes.sparsity_detector(ad) isa DI.DenseSparsityDetector
_uses_ad_sparsity_detector(_) = false

"""
    maybe_unwrap_prob_for_enzyme(prob, autodiffs...)

If the problem function is wrapped by AutoSpecialize and any of the given AD backends
use Enzyme, return a copy of `prob` with the unwrapped raw function. Otherwise return
`prob` unchanged.

This should be called early in `__init` so that all downstream AD-related constructions
(Jacobian cache, trust region operators, linesearch, forcing) receive the unwrapped problem.
"""
function maybe_unwrap_prob_for_enzyme(prob, autodiffs...)
    f = prob.f.f
    is_fw_wrapped(f) || return prob
    # For the opaque path, Enzyme cannot differentiate through the OpaqueParams
    # byte unpacking, so fully revert: raw residual + the concrete `p` unpacked
    # from the opaque container.
    if f isa AutoDePSpecializeCallable
        for ad in autodiffs
            if _uses_enzyme_ad(ad)
                P = _autodep_paramtype(f)
                rawp = RespecializeParams.unpack(prob.p, P)
                prob = @set prob.f.f = f.orig
                return @set prob.p = rawp
            end
        end
        return prob
    end
    for ad in autodiffs
        _uses_enzyme_ad(ad) && return @set prob.f.f = get_raw_f(f)
    end
    return prob
end

# Default dispatch assumes no ForwardDiff loaded.
# The ForwardDiff extension overrides these with dual-aware versions.

function wrapfun_iip(ff, inputs::Tuple)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(ff), (typeof(inputs),), (Nothing,)
    )
end

"""
    standardize_forwarddiff_tag(ad, prob)

If the user function was wrapped via AutoSpecialize (`FunctionWrappersWrapper`),
stamp `ad` with `NonlinearSolveTag` so that the dual numbers generated during
Jacobian computation match the precompiled wrapper type signatures.

For all other AD backends, or for problems whose function was not wrapped
(FullSpecialize, OOP, or non-in-place), returns `ad` unchanged so
DifferentiationInterface generates a runtime tag from the function type.

The `AutoForwardDiff`-specific dispatch is provided by the ForwardDiff extension.
"""
standardize_forwarddiff_tag(ad, prob) = ad

"""
    maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)

Attempt to wrap an in-place problem function with `FunctionWrappersWrapper` for the
norecompile (AutoSpecialize) pathway. Returns an `AutoSpecializeCallable` wrapping both
the `FunctionWrappersWrapper` and the original function if the problem is IIP with
array-typed state, non-dual eltype, and has opted in to `AutoSpecialize`; otherwise
returns the original function unchanged.

OOP functions are not wrapped because guessing the return type is unreliable.
Non-array state (e.g. scalar `Number` u0) is not wrapped because the Dual-aware
`wrapfun_iip` signatures only cover array state. Dual-eltype state is not wrapped
because `promote_u0` upgrades `u0` to a `Dual`-eltype array whenever the user's
outer-AD pass injected duals into `p`; in that case the wrapper's signatures would
be keyed off the outer Dual tag and miss the inner value-typed dispatch that the
forward-diff extension builds via `Utils.value` / `nodual_value`.
"""
function maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)
    u0 = prob.u0
    p = prob.p

    # Skip wrapping when called from inside Enzyme reverse-mode AD. The
    # FunctionWrappersWrapper construction emits ptrtoint/store patterns that
    # defeat Enzyme's static activity analysis (and `set_runtime_activity` is
    # not sufficient to recover correctness), so the wrapper must not be
    # constructed on the outer-AD path. The unwrapped function works fine.
    EnzymeCore.within_autodiff() && return prob.f.f

    # Already wrapped — idempotent
    is_fw_wrapped(prob.f.f) && return prob.f.f

    # Only wrap IIP functions. OOP wrapping requires guessing the return type,
    # which doesn't always work (see DiffEqBase for precedent).
    SciMLBase.isinplace(prob) || return prob.f.f

    # Only wrap array-typed state. The ForwardDiff-aware `wrapfun_iip` dispatches
    # on `AbstractArray` state and builds signatures over `similar(u0, ::DualT)`.
    u0 isa AbstractArray || return prob.f.f

    # Skip wrapping when `u0` already carries a Dual eltype — `promote_u0` does
    # this whenever outer-AD injected duals into `p`. The wrapper's signatures
    # would then be keyed off the outer Dual tag and miss the inner value-typed
    # dispatch that the forward-diff extension builds via `Utils.value` /
    # `nodual_value`.
    SciMLBase.isdualtype(eltype(u0)) && return prob.f.f

    # Only wrap when AutoSpecialize (the default) or AutoDePSpecialize is active.
    # FullSpecialize opts out of wrapping, keeping the exact function type.
    # (The opaque-`p` sub-case of AutoDePSpecialize is handled earlier in
    # `maybe_wrap_f` via `maybe_opaque_wrap`; reaching here under
    # AutoDePSpecialize means `p` was not opaque-ified, so wrap like
    # AutoSpecialize.)
    spec = SciMLBase.specialization(prob.f)
    (spec === SciMLBase.AutoSpecialize || spec === SciMLBase.AutoDePSpecialize) ||
        return prob.f.f

    orig = prob.f.f
    inputs = (u0, u0, p)
    return AutoSpecializeCallable(wrapfun_iip(orig, inputs), orig)
end

# ---------------------------------------------------------------------------
# AutoDePSpecialize: de-specialize the parameter object.
#
# When `p` is an `isbits` non-`NullParameters` payload, pack it into a
# `RespecializeParams.OpaqueParams` container and wrap the residual so its
# `FunctionWrappersWrapper` signature carries `OpaqueParams` in the `p` slot
# (via `RespecializeParams.OpaqueVoid`). The concretized problem type — and the
# solver compilation — becomes independent of the user's parameter struct type,
# so one precompiled solve is shared across all such types, while the residual
# still runs fully specialized on `p` (recovered by a type-stable,
# allocation-free unpack). Mirrors DiffEqBase's AutoDePSpecialize path; the
# marker and the mechanism (`OpaqueVoid`, `wrap_void_opaque`) are shared with
# it via SciMLBase and RespecializeParams.
# ---------------------------------------------------------------------------

"""
    should_opaque_p(p) -> Bool

Policy for whether the AutoDePSpecialize path de-specializes `p`: `true` for any
payload except a `NullParameters` sentinel or an already-packed container.
`isbits` payloads pack into a `RespecializeParams.OpaqueParams` (byte copy),
everything else into a `RespecializeParams.OpaqueRef` (by reference). Defined
locally rather than taken from SciMLBase so this does not depend on an unreleased
SciMLBase symbol.

De-specializing makes the concretized `prob.p` an opaque container, so the level
is a forward-solve latency tool: reverse-mode AD of `p` and symbolic parameter
indexing read the concrete parameter and are not supported (reverse-mode support
is a planned follow-up). Recover the payload with
`RespecializeParams.unpack(sol.prob.p, typeof(p))`.
"""
@inline should_opaque_p(p) = !(p isa SciMLBase.NullParameters) &&
    !(p isa RespecializeParams.OpaqueParams) && !(p isa RespecializeParams.OpaqueRef)

"""
    AutoDePSpecializeCallable{FW}

Like [`AutoSpecializeCallable`](@ref) but for the AutoDePSpecialize path: `fw` is
a `FunctionWrappersWrapper` whose signatures carry `RespecializeParams.OpaqueParams`
in the parameter slot, `orig` is the raw user residual, and `ptype` records the
concrete parameter type so fallback paths can unpack the opaque container back to
it. Both `orig` and `ptype` are type-erased fields (not type parameters) so the
callable's Julia type is `FW` only — problems whose parameter struct types differ
share the same `AutoDePSpecializeCallable{FW}`, which is the whole point.
"""
struct AutoDePSpecializeCallable{FW}
    fw::FW
    orig::Any     # type-erased so all wrapped residuals share one Julia type
    ptype::DataType
end

@inline (f::AutoDePSpecializeCallable)(args...) = f.fw(args...)

is_fw_wrapped(::AutoDePSpecializeCallable) = true

# A non-FunctionWrapper callable that still accepts the OpaqueParams `p`: it
# unpacks to the concrete `ptype` and forwards to the raw residual. Used by
# AD/DI fallbacks that need a differentiable callable rather than the wrapper.
get_raw_f(f::AutoDePSpecializeCallable) = RespecializeParams.OpaqueVoid(f.ptype, f.orig)

_autodep_paramtype(f::AutoDePSpecializeCallable) = f.ptype

# Base (no ForwardDiff) opaque wrapper: single-signature. The ForwardDiff
# extension overrides this with the dual-aware variants.
function wrapfun_iip_opaque(ff, ::Type{P}, inputs::Tuple) where {P}
    sig = Tuple{typeof(inputs[1]), typeof(inputs[2]), typeof(inputs[3])}
    return RespecializeParams.wrap_void_opaque(ff, P, (sig,))
end

"""
    maybe_opaque_wrap(prob) -> prob or nothing

If `prob` opts into `AutoDePSpecialize` and its `p` is opaque-able, return a copy
of `prob` with the residual wrapped through `OpaqueVoid` and `p` packed into an
`OpaqueParams`. Otherwise return `nothing` (the caller then falls back to the
plain [`maybe_wrap_nonlinear_f`](@ref) path).

Scoped to in-place `NonlinearProblem`/`ImmutableNonlinearProblem` with array,
non-dual state, outside Enzyme AD. `NonlinearLeastSquaresProblem` and the
Enzyme/dual paths keep the existing behavior; recovering the original `p` from a
solved problem's `sol.prob.p` is `RespecializeParams.unpack(sol.prob.p, P)`.
"""
function maybe_opaque_wrap(prob::AbstractNonlinearProblem)
    SciMLBase.specialization(prob.f) === SciMLBase.AutoDePSpecialize || return nothing
    EnzymeCore.within_autodiff() && return nothing
    is_fw_wrapped(prob.f.f) && return nothing
    (prob isa NonlinearProblem || prob isa SciMLBase.ImmutableNonlinearProblem) ||
        return nothing
    SciMLBase.isinplace(prob) || return nothing

    u0 = prob.u0
    p = prob.p
    u0 isa AbstractArray || return nothing
    SciMLBase.isdualtype(eltype(u0)) && return nothing
    should_opaque_p(p) || return nothing

    P = typeof(p)
    orig = prob.f.f
    fw = wrapfun_iip_opaque(orig, P, (u0, u0, p))
    newprob = @set prob.f.f = AutoDePSpecializeCallable{typeof(fw)}(fw, orig, P)
    @set! newprob.p = RespecializeParams.pack_auto(p)
    return newprob
end
