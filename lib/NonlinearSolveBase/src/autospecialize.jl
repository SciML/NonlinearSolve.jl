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
(AutoSpecialize) pathway. Returns the wrapped function if conditions are met (concrete
`Vector{Float64}` state and `Vector{Float64}` or `NullParameters` parameters), otherwise
returns the original function.

This enables precompilation of the solver code for standard argument types, avoiding
recompilation for each unique user function type.
"""
function maybe_wrap_nonlinear_f(prob::AbstractNonlinearProblem)
    u0 = prob.u0
    p = prob.p

    # Only wrap for standard types
    u0 isa Vector{Float64} || return prob.f.f
    (p isa Vector{Float64} || p isa SciMLBase.NullParameters) || return prob.f.f

    if SciMLBase.isinplace(prob)
        inputs = (u0, u0, p)
        return wrapfun_iip(prob.f.f, inputs)
    else
        inputs = (u0, p)
        return wrapfun_oop(prob.f.f, inputs)
    end
end
