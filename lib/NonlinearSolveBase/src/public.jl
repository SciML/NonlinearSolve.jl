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
    solve_cache!(cache::AbstractNonlinearSolveCache; step_observer = nothing) -> ReturnCode

Drive an initialized stepping cache to termination without constructing a
`SciMLBase.NonlinearSolution`.

This allocation-sensitive interface is intended for nested solvers that already own a
cache from `init`. It is available only when the cache implements the nonlinear solver
iterator interface. Unlike `solve!(cache)`, it does not transform a bounded problem's
internal unconstrained state back to bounded coordinates.

# Arguments

- `cache::AbstractNonlinearSolveCache`: an initialized cache with an
  `InternalAPI.step!` implementation.

# Keywords

- `step_observer = nothing`: an optional callable invoked after each nonlinear step as
  `step_observer(u, fu, iteration)`. The `u` and `fu` arguments alias the cache and must
  not be mutated.

# Returns

The final `SciMLBase.ReturnCode`. The final state remains available through
`SymbolicIndexingInterface.state_values(cache)`.

# Throws

`ArgumentError` if `cache` does not support the stepping interface.

# Extension Rules

Implement `NonlinearSolveBase.InternalAPI.step!` on a cache before calling this function.
Use `solve!(cache)` for [`NonlinearSolveNoInitCache`](@ref), which intentionally has no
stepping state.

# Examples

```julia
import NonlinearSolve
import NonlinearSolveBase

prob = NonlinearSolve.NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
cache = NonlinearSolve.init(prob, NonlinearSolve.NewtonRaphson())
retcode = NonlinearSolveBase.solve_cache!(cache)
```
"""
function solve_cache! end

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

"""
    termination_condition_result(cache, fallback_u, fallback_t, solver_retcode) -> (u, t, retcode)

Return the state, time or iteration marker, and return code selected by a nonlinear
termination cache.

This developer API is for solver packages that drive an
`AbstractNonlinearTerminationMode` cache outside NonlinearSolve's standard solve loop.
It applies the cache's termination policy without exposing cache storage. Safe-best modes
return their recorded best state and saved marker when one is available; other modes
return `fallback_u` and `fallback_t`.

# Arguments

  - `cache`: A cache returned by `SciMLBase.init` for a public nonlinear termination mode.
  - `fallback_u`: Final state produced by the enclosing solver.
  - `fallback_t`: Time or iteration marker associated with `fallback_u`.
  - `solver_retcode`: Unmodified return code produced by the enclosing solver.

# Returns

  - `u`: The selected state. A safe-best cache returns a copy of its retained best state
    when available; otherwise this is `fallback_u`.
  - `t`: Marker associated with `u`. A safe-best cache returns its saved marker when
    available; otherwise this is `fallback_t`.
  - `retcode`: The cache-adjusted solver return code. A standard termination event maps
    to `ReturnCode.Success`; safe modes preserve a more specific cached result.

# Developer Contract

Call this only after the final update of a cache returned by
`SciMLBase.init(prob, termination_condition, du, u; kwargs...)`. The fallback values must
describe the enclosing solver's actual final result. This is a consumer API, not an
extension point: solver packages must not extend it or access the cache's fields directly.

# Example

```julia
using NonlinearSolveBase, SciMLBase

prob = NonlinearProblem((u, p) -> u, [1.0])
cache = init(prob, AbsNormTerminationMode(NonlinearSolveBase.Linf_NORM), [1.0], [1.0])
termination_condition_result(cache, [0.0], 1.0, ReturnCode.Terminated)
```
"""
function termination_condition_result end

@compat(public, (termination_condition_result,))

abstract type AbstractSafeBestNonlinearTerminationMode <:
AbstractSafeNonlinearTerminationMode end

#! format: off
const TERM_DOCS = Dict(
    :Norm => doc"``\| r \| ≤ reltol × \| r + u \|`` or ``\| r \| ≤ abstol``",
    :Rel => doc"``\mathrm{all} \left(| r_i | ≤ reltol × | r_i + u_i | \right)``",
    :RelNorm => doc"``\| r \| ≤ reltol × \| r + u \|``",
    :Abs => doc"``\mathrm{all} \left( | r_i | ≤ abstol \right)``",
    :AbsNorm => doc"``\| r \| ≤ abstol``"
)

const TERM_INTERNALNORM_DOCS = """
where `internalnorm` is a callable norm to use for the termination condition. Special handling
is done for `norm(_, 2)`, `norm`, `norm(_, Inf)`, and `maximum(abs, _)`."""

const TERM_INTERFACE_DOCS = """
The nonlinear solver evaluates this mode through a termination cache. The cache is called as
`cache(du, u, uprev)`, where `du` is the current residual `f(u)`, `u` is the current iterate,
and `uprev` is the previous accepted iterate. The solver supplies `abstol` and `reltol` when
the cache is initialized. The name `du` is retained for compatibility with the step-based
termination interface in DifferentialEquations.jl; it is not a Newton increment here.

# Arguments

- `du`: Current residual `f(u)`.
- `u`: Current iterate.
- `uprev`: Previous accepted iterate. Plain modes ignore it; safe modes use it only when
  `max_stalled_steps` is enabled.

# Returns

- `Bool`: `true` when the mode's convergence or safety criterion requests termination, and
  `false` when the solver should continue. The solver records `ReturnCode.Success` for a
  tolerance-based termination. Safe modes can instead record `ReturnCode.Unstable`,
  `ReturnCode.Stalled`, or `ReturnCode.StalledSuccess` when a safety criterion fires.
"""

const TERM_RELATIVE_WARNING = """
!!! warning

    The relative criterion compares the residual with `r + u`, rather than with a step and the
    next iterate. This is the behavior currently implemented by NonlinearSolve. It can produce
    a misleading relative measure when the residual and iterate have very different scales.
    Prefer an absolute mode when this behavior is not appropriate. See
    [#1149](https://github.com/SciML/NonlinearSolve.jl/issues/1149).
"""

const TERM_EXAMPLE = """
# Examples

```julia
using NonlinearSolve, NonlinearSolveBase

prob = NonlinearProblem((u, p) -> [u[1]^2 - 2], [1.0])
sol = solve(prob, NewtonRaphson(); termination_condition = MODE)
```
"""

const TERM_NORM_ARGS = """
# Constructor Arguments

- `internalnorm`: Callable used to reduce the residual and the residual-plus-iterate pair to
  scalar objectives. It may be `norm`, `norm(_, 2)`, `norm(_, Inf)`, `maximum(abs, _)`, or a
  custom callable with the corresponding inputs.
"""

const TERM_SAFE_KEYWORDS = """
# Keywords

- `protective_threshold`: Optional multiplier for the initial objective. When set, a current
  objective larger than `protective_threshold * initial_objective * length(du)` returns
  `ReturnCode.Unstable`. Defaults to `nothing`, which disables this check.
- `patience_steps::Int`: Number of objective evaluations retained for the no-improvement check.
  Defaults to `100`.
- `patience_objective_multiplier`: Enables the no-improvement check only while the current
  objective is at most this multiple of the requested tolerance. Defaults to `3`.
- `min_max_factor`: Declares the objective stalled when the minimum objective in the retained
  history is less than this factor times the maximum. Defaults to `1.3`.
- `max_stalled_steps`: Optional number of steps after which a step-size stall is checked.
  `nothing` disables this additional check. Defaults to `nothing`.
"""

const TERM_SAFE_SEMANTICS = """
# Termination semantics

In addition to the base tolerance criterion, safe modes terminate when the objective is not
improving under the configured patience settings, when the objective is nonfinite, or when an
enabled protective or step-stall check fires. A non-least-squares problem receives
`ReturnCode.Stalled` for an objective or step stall; a least-squares problem receives
`ReturnCode.StalledSuccess`. A `SafeBest` mode retains the iterate with the best objective seen
so far and returns that iterate when the termination cache is used to build the solution.
"""
#! format: on

for name in (:Rel, :Abs)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]
    criterion = name == :Rel ? TERM_RELATIVE_WARNING : ""
    example = replace(TERM_EXAMPLE, "MODE" => "$(struct_name)()")

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        Here `r` denotes the residual `f(u)` supplied by the nonlinear solver and `u` denotes
        the current iterate. This is a residual criterion, not a Newton-step criterion.

        $($TERM_INTERFACE_DOCS)

        # Constructor

            $($struct_name)()

        This constructor takes no arguments.

        $($criterion)

        $($example)
        """
        struct $(struct_name) <: AbstractNonlinearTerminationMode end
    end
end

for name in (:Norm, :RelNorm, :AbsNorm)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]
    criterion = name == :RelNorm ? TERM_RELATIVE_WARNING : ""
    example = replace(
        TERM_EXAMPLE,
        "MODE" => "$(struct_name)(Base.Fix1(maximum, abs))"
    )

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        Here `r` denotes the residual `f(u)` supplied by the nonlinear solver and `u` denotes
        the current iterate. This is a residual criterion, not a Newton-step criterion.

        $($TERM_INTERFACE_DOCS)

        ## Constructor

            $($struct_name)(internalnorm)

        $($TERM_INTERNALNORM_DOCS)

        $($TERM_NORM_ARGS)

        $($criterion)

        $($example)
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
    criterion = norm_type == :AbsNorm ?
        "The base criterion is ``\\|r\\| ≤ abstol``." :
        "The base criterion is ``\\|r\\| / (\\|r + u\\| + \\epsilon(reltol)) ≤ reltol``."
    example = replace(
        TERM_EXAMPLE,
        "MODE" => "$(struct_name)(Base.Fix1(maximum, abs); max_stalled_steps = 10)"
    )

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

        $($criterion)

        $($TERM_INTERFACE_DOCS)

        ## Constructor

            $($struct_name)(
                internalnorm; protective_threshold = nothing,
                patience_steps = 100, patience_objective_multiplier = 3,
                min_max_factor = 1.3, max_stalled_steps = nothing
            )

        $($TERM_INTERNALNORM_DOCS)

        $($TERM_NORM_ARGS)

        $($TERM_SAFE_KEYWORDS)

        $($TERM_SAFE_SEMANTICS)

        $($example)
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
