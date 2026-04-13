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
    maybe_unwrap_prob_for_enzyme(prob, autodiffs...)

If the problem function is wrapped by AutoSpecialize and any of the given AD backends
use Enzyme, return a copy of `prob` with the unwrapped raw function. Otherwise return
`prob` unchanged.

This should be called early in `__init` so that all downstream AD-related constructions
(Jacobian cache, trust region operators, linesearch, forcing) receive the unwrapped problem.
"""
function maybe_unwrap_prob_for_enzyme(prob, autodiffs...)
    is_fw_wrapped(prob.f.f) || return prob
    for ad in autodiffs
        _uses_enzyme_ad(ad) && return @set prob.f.f = get_raw_f(prob.f.f)
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
`Vector{Float64}` state and has opted in to `AutoSpecialize`; otherwise returns the
original function unchanged.

OOP functions are not wrapped because guessing the return type is unreliable.
"""
function maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)
    u0 = prob.u0
    p = prob.p

    # Already wrapped — idempotent
    is_fw_wrapped(prob.f.f) && return prob.f.f

    # Only wrap for Vector{Float64} state. The FunctionWrappersWrapper signatures
    # built by `wrapfun_iip` are keyed off `eltype(u0)`, and `nonlinearsolve_forwarddiff_solve`
    # calls `remake(prob; u0=Utils.value(u0), p=Utils.value(p))` to build a Float64 inner
    # problem that reuses the already-wrapped `f.f`. If the outer u0 was promoted to a Dual
    # eltype (e.g. by `promote_u0` in `get_concrete_problem` when `p` carries outer-AD
    # duals), the stored wrapper would have Dual-eltype signatures and the inner Float64
    # solve would fail with "No matching function wrapper" when the Jacobian cache calls
    # `f(du::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})`. Restricting
    # wrapping to Vector{Float64} keeps the wrapper signatures aligned with the inner
    # solve and with the precompiled FWW specializations.
    u0 isa Vector{Float64} || return prob.f.f

    # Only wrap IIP functions. OOP wrapping requires guessing the return type,
    # which doesn't always work (see DiffEqBase for precedent).
    SciMLBase.isinplace(prob) || return prob.f.f

    # Only wrap when AutoSpecialize is active (the default).
    # FullSpecialize opts out of wrapping, keeping the exact function type.
    SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize || return prob.f.f

    orig = prob.f.f
    inputs = (u0, u0, p)
    return AutoSpecializeCallable(wrapfun_iip(orig, inputs), orig)
end
