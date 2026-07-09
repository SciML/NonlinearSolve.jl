"""
    UNITLESS_ABS2(x)

Return a unitless squared magnitude for `x`.

This developer API is used by nonlinear solver internals to compare residuals, steps, and
state values without preserving physical units. Numbers use `abs2`; arrays and nested
SciML array containers reduce over their stored values.

### Arguments

  - `x`: A number, array, `AbstractVectorOfArray`, or `ArrayPartition`.

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.UNITLESS_ABS2([3.0, 4.0])
```
"""
function UNITLESS_ABS2 end

"""
    NAN_CHECK(x)::Bool

Return `true` when `x` or any value stored in `x` is `NaN`.

This developer API is used by solver implementations before accepting iterates and
residuals.

### Arguments

  - `x`: A scalar or array-like value to inspect.

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.NAN_CHECK([1.0, NaN])
```
"""
function NAN_CHECK end

"""
    L2_NORM(u)

Compute the Euclidean norm used by NonlinearSolve internals.

The implementation has fast paths for numbers, dense arrays, and static arrays, and falls
back to `norm(u, 2)` for other array-like states.

### Arguments

  - `u`: A scalar or array-like state.

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.L2_NORM([3.0, 4.0])
```
"""
function L2_NORM end

"""
    Linf_NORM(u)

Compute the infinity norm used by NonlinearSolve internals.

### Arguments

  - `u`: A scalar or array-like state.

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.Linf_NORM([-3.0, 4.0])
```
"""
function Linf_NORM end

"""
    get_tolerance([u], η, ::Type{T})

Convert or choose a nonlinear solver tolerance of real type `T`.

When `η === nothing`, NonlinearSolve chooses a default tolerance based on `T` and, for
array-free scalar/static states, uses a GPU-compatible exponent path.

### Arguments

  - `u`: Optional state value used by some specialization paths.
  - `η`: User-provided tolerance or `nothing`.
  - `T`: Target numeric type.

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.get_tolerance(nothing, Float64)
NonlinearSolveBase.get_tolerance([1.0], nothing, Float64)
```
"""
function get_tolerance end

# Forward declarations of functions for forward mode AD
"""
    nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)

Solve `prob` through a ForwardDiff-aware wrapper and return the primal solution together
with parameter partials.

This is developer API for solver packages that need to propagate dual-number sensitivities
through specialized nonlinear solve implementations.

### Arguments

  - `prob`: A SciML nonlinear problem whose parameters may carry ForwardDiff dual values.
  - `alg`: The nonlinear solver algorithm.
  - `args...`: Additional positional arguments forwarded to `solve`.

### Keyword Arguments

All keyword arguments are forwarded to the underlying nonlinear solve.

### Returns

A pair `(sol, partials)` where `sol` is the primal nonlinear solution and `partials`
contains the propagated parameter partials.
"""
function nonlinearsolve_forwarddiff_solve end

"""
    nonlinearsolve_dual_solution(u, partials, p)

Reconstruct a dual-valued nonlinear solution from a primal state and parameter partials.

This is developer API paired with [`nonlinearsolve_forwarddiff_solve`](@ref).

### Arguments

  - `u`: The primal nonlinear solution state.
  - `partials`: The partial derivatives returned by the ForwardDiff solve path.
  - `p`: Original parameter value, used to recover the dual tag and partial layout.
"""
function nonlinearsolve_dual_solution end
function nonlinearsolve_∂f_∂p end
function nonlinearsolve_∂f_∂u end
function nlls_generate_vjp_function end
function nodual_value end

"""
    pickchunksize(x) = pickchunksize(length(x))
    pickchunksize(x::Int)

Determine the chunk size for ForwardDiff and PolyesterForwardDiff based on the input length.
"""
function pickchunksize end

"""
    AbstractNonlinearTerminationMode

Abstract supertype for nonlinear solver termination modes.

Concrete subtypes define how an update `Δu`, current iterate `u`, and tolerances are
combined to decide whether a nonlinear solve has converged.

See also [`RelTerminationMode`](@ref), [`AbsTerminationMode`](@ref),
[`NormTerminationMode`](@ref), [`RelNormTerminationMode`](@ref), and
[`AbsNormTerminationMode`](@ref).
"""
abstract type AbstractNonlinearTerminationMode end

"""
    AbstractSafeNonlinearTerminationMode <: AbstractNonlinearTerminationMode

Abstract supertype for termination modes that include stagnation or divergence safeguards.

Safe termination modes preserve the usual tolerance check while also stopping solves that
stop improving according to the mode-specific objective history.

See also [`RelNormSafeTerminationMode`](@ref), [`AbsNormSafeTerminationMode`](@ref),
[`RelNormSafeBestTerminationMode`](@ref), and [`AbsNormSafeBestTerminationMode`](@ref).
"""
abstract type AbstractSafeNonlinearTerminationMode <: AbstractNonlinearTerminationMode end
abstract type AbstractSafeBestNonlinearTerminationMode <:
AbstractSafeNonlinearTerminationMode end

#! format: off
const TERM_DOCS = Dict(
    :Norm => doc"``\| Δu \| ≤ reltol × \| Δu + u \|`` or ``\| Δu \| ≤ abstol``",
    :Rel => doc"``\mathrm{all} \left(| Δu | ≤ reltol × | u | \right)``",
    :RelNorm => doc"``\| Δu \| ≤ reltol × \| Δu + u \|``",
    :Abs => doc"``\mathrm{all} \left( | Δu | ≤ abstol \right)``",
    :AbsNorm => doc"``\| Δu \| ≤ abstol``"
)

const TERM_INTERNALNORM_DOCS = """
where `internalnorm` is the norm to use for the termination condition. Special handling is
done for `norm(_, 2)`, `norm`, `norm(_, Inf)`, and `maximum(abs, _)`"""
#! format: on

for name in (:Rel, :Abs)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the nonlinear solver and ``u`` denotes the solution.
        """
        struct $(struct_name) <: AbstractNonlinearTerminationMode end
    end
end

for name in (:Norm, :RelNorm, :AbsNorm)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the inner nonlinear solver.

        ## Constructor

            $($struct_name)(internalnorm = nothing)

        $($TERM_INTERNALNORM_DOCS).
        """
        struct $(struct_name){F} <: AbstractNonlinearTerminationMode
            internalnorm::F

            function $(struct_name)(internalnorm::F) where {F}
                norm = Utils.standardize_norm(internalnorm)
                return new{typeof(norm)}(norm)
            end
        end
    end
end

for norm_type in (:RelNorm, :AbsNorm), safety in (:Safe, :SafeBest)

    struct_name = Symbol(norm_type, safety, :TerminationMode)
    supertype_name = Symbol(:Abstract, safety, :NonlinearTerminationMode)

    doctring = safety == :Safe ?
        "Essentially [`$(norm_type)TerminationMode`](@ref) + terminate if there \
                has been no improvement for the last `patience_steps` + terminate if the \
                solution blows up (diverges)." :
        "Essentially [`$(norm_type)SafeTerminationMode`](@ref), but caches the best\
                solution found so far."

    @eval begin
        """
            $($struct_name) <: $($supertype_name)

        $($doctring)

        ## Constructor

            $($struct_name)(
                internalnorm; protective_threshold = nothing,
                patience_steps = 100, patience_objective_multiplier = 3,
                min_max_factor = 1.3, max_stalled_steps = nothing
            )

        $($TERM_INTERNALNORM_DOCS).
        """
        @concrete struct $(struct_name) <: $(supertype_name)
            internalnorm
            protective_threshold
            patience_steps::Int
            patience_objective_multiplier
            min_max_factor
            max_stalled_steps <: Union{Nothing, Int}

            function $(struct_name)(
                    internalnorm::F; protective_threshold = nothing,
                    patience_steps = 100, patience_objective_multiplier = 3,
                    min_max_factor = 1.3, max_stalled_steps = nothing
                ) where {F}
                norm = Utils.standardize_norm(internalnorm)
                return new{
                    typeof(norm), typeof(protective_threshold),
                    typeof(patience_objective_multiplier),
                    typeof(min_max_factor), typeof(max_stalled_steps),
                }(
                    norm, protective_threshold, patience_steps,
                    patience_objective_multiplier, min_max_factor, max_stalled_steps
                )
            end
        end
    end
end
