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
struct AutoSpecializeCallable{FW} <: Function
    fw::FW
    orig::Any  # type-erased: all wrapped functions share the same Julia type
end

# FunctionWrappersWrapper throws NoFunctionWrapperFoundError (or ErrorException
# in older versions) when no signature matches. Fall back to the original function
# for unsupported argument types (e.g., external packages like LeastSquaresOptim
# doing their own ForwardDiff with different tags/chunksizes, or JVP paths that
# bypass tag standardization).
function (f::AutoSpecializeCallable)(args...)
    try
        return f.fw(args...)
    catch e
        if e isa FunctionWrappersWrappers.NoFunctionWrapperFoundError ||
                (e isa ErrorException && contains(e.msg, "No matching function wrapper"))
            return f.orig(args...)
        end
        rethrow()
    end
end

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

# Default dispatch assumes no ForwardDiff loaded.
# The ForwardDiff extension overrides these with dual-aware versions.

function wrapfun_iip(ff, inputs::Tuple)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(ff), (typeof(inputs),), (Nothing,)
    )
end

function wrapfun_oop(ff, inputs::Tuple)
    return FunctionWrappersWrappers.FunctionWrappersWrapper(
        ff, (typeof(inputs),), (typeof(inputs[1]),)
    )
end

"""
    standardize_forwarddiff_tag(ad, prob)

If `ad` is an `AutoForwardDiff` with no custom tag and `prob` has `Vector{Float64}` state,
stamp it with `NonlinearSolveTag` so that the dual numbers generated during Jacobian
computation match the precompiled `FunctionWrappersWrapper` type signatures.

For all other AD backends or non-standard argument types, returns `ad` unchanged.

The `AutoForwardDiff`-specific dispatch is provided by the ForwardDiff extension.
"""
standardize_forwarddiff_tag(ad, prob) = ad

"""
    maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)

Attempt to wrap the problem function with `FunctionWrappersWrapper` for the norecompile
(AutoSpecialize) pathway. Returns an `AutoSpecializeCallable` wrapping both the
`FunctionWrappersWrapper` and the original function if the state is concrete
`Vector{Float64}`, otherwise returns the original function. The precompiled wrappers
cover `Vector{Float64}` and `NullParameters` parameter types, but any parameter type
is accepted — mismatches fall back through the `AutoSpecializeCallable` try-catch path.
"""
function maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)
    u0 = prob.u0
    p = prob.p

    # Only wrap for Vector{Float64} state
    u0 isa Vector{Float64} || return prob.f.f

    # Already wrapped — idempotent
    is_fw_wrapped(prob.f.f) && return prob.f.f

    orig = prob.f.f
    if SciMLBase.isinplace(prob)
        inputs = (u0, u0, p)
        return AutoSpecializeCallable(wrapfun_iip(orig, inputs), orig)
    else
        # OOP NonlinearLeastSquaresProblem: the return type (residual) may differ from u0
        # (e.g. Matrix, or Vector of different length). Skip wrapping since the
        # FunctionWrappersWrapper return type is based on typeof(u0).
        if prob isa SciMLBase.NonlinearLeastSquaresProblem
            return prob.f.f
        end
        inputs = (u0, p)
        return AutoSpecializeCallable(wrapfun_oop(orig, inputs), orig)
    end
end
