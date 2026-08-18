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
    solve_cache!(cache; step_observer = nothing) -> ReturnCode

Drive an initialized nonlinear solver cache to termination without constructing a
`NonlinearSolution`.

This allocation-sensitive interface is intended for nested solvers that already own a
cache from `init` and whose algorithm supports the nonlinear solver iterator interface.
`step_observer`, when provided, is called after every nonlinear iteration as
`step_observer(u, fu, iteration)`. The state and residual arguments alias the solver
cache and must not be mutated. The returned `SciMLBase.ReturnCode` reports the final
solver status; the final state remains available through
`SymbolicIndexingInterface.state_values(cache)`.

Unlike `solve!(cache)`, this function does not transform a bounded problem's internal
unconstrained state back to the bounded coordinates.
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
    :Norm => doc"``\| Δu \| ≤ reltol × \| Δu + u \|`` or ``\| Δu \| ≤ abstol``",
    :Rel => doc"``\mathrm{all} \left(| Δu | ≤ reltol × | Δu + u | \right)``",
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
                min_max_factor = 1.3, max_stalled_steps = nothing, gtol = nothing
            )

        $($TERM_INTERNALNORM_DOCS).

        `gtol` enables the gradient-stationarity criterion described in
        [`gradient_stationarity_measure`](@ref). It is disabled by default; see
        [`check_gradient_and_update!`](@ref) for how a solver supplies the Jacobian and
        for the differing interpretation on square and least-squares problems.
        """
        @concrete struct $(struct_name) <: $(supertype_name)
            internalnorm
            protective_threshold
            patience_steps::Int
            patience_objective_multiplier
            min_max_factor
            max_stalled_steps <: Union{Nothing, Int}
            gtol <: Union{Nothing, Real}

            function $(struct_name)(
                    internalnorm::F; protective_threshold = nothing,
                    patience_steps = 100, patience_objective_multiplier = 3,
                    min_max_factor = 1.3, max_stalled_steps = nothing, gtol = nothing
                ) where {F}
                norm = Utils.standardize_norm(internalnorm)
                return new{
                    typeof(norm), typeof(protective_threshold),
                    typeof(patience_objective_multiplier),
                    typeof(min_max_factor), typeof(max_stalled_steps), typeof(gtol),
                }(
                    norm, protective_threshold, patience_steps,
                    patience_objective_multiplier, min_max_factor, max_stalled_steps, gtol
                )
            end
        end
    end
end

"""
    gradient_stationarity_measure(J, fu) -> Union{Nothing, Real}

Return the scale-free stationarity measure

```math
\\max_j \\frac{|(J^\\top F)_j|}{\\|J_j\\|_2 \\, \\|F\\|_2}
```

where ``J_j`` is the `j`-th column of `J`. This is the cosine of the angle between the
residual and that column, so it is the criterion MINPACK exposes as `gtol` and the one
Ceres and `scipy.optimize.least_squares` use.

A least-squares solution satisfies ``J^\\top F = 0``, not ``F = 0``, so a residual test
alone cannot certify convergence on a problem whose residual is nonzero at the optimum.
Testing ``\\|J^\\top F\\|`` directly would not do either: it carries units of residual per
unit of `u` and therefore moves under a rescaling of either. Dividing by
``\\|J_j\\| \\|F\\|`` removes both, leaving a dimensionless quantity invariant under
`u -> S u` for invertible diagonal `S`, under `F -> c F` for scalar `c`, and under an
orthogonal transformation of the residual.

# Arguments

  - `J`: Jacobian at the current iterate. Must describe the same iterate as `fu`.
  - `fu`: Residual at the current iterate.

# Returns

The measure, or `nothing` when it is not defined: `‖F‖ = 0` (the residual test already
covers that case, and the ratio is `0/0`) or a non-finite residual.

# Examples

```julia
using NonlinearSolveBase

J = [1.0 0.0; 0.0 1.0]
NonlinearSolveBase.gradient_stationarity_measure(J, [0.0, 1.0])
```
"""
function gradient_stationarity_measure end

"""
    gradient_measure_supported(J) -> Bool

Return whether [`gradient_stationarity_measure`](@ref) can be evaluated for a Jacobian
representation `J`.

The measure needs the individual column norms of `J`, so it is defined for stored matrices
and for scalars, and undefined for a matrix-free operator. A solver that holds only an
operator, or an approximation such as the secant Jacobian a quasi-Newton method carries,
should not reach the gradient criterion at all; this predicate is what makes that
degradation silent rather than an error.

# Arguments

  - `J`: A Jacobian representation held by a solver cache.

# Returns

`true` for `AbstractMatrix` and `Number`, `false` otherwise.

# Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.gradient_measure_supported([1.0 0.0; 0.0 1.0]) # true
```
"""
function gradient_measure_supported end

"""
    check_gradient_and_update!(cache, J, fu, u) -> Bool

Apply the gradient-stationarity criterion of [`gradient_stationarity_measure`](@ref) to a
solver cache and stop the solve when it is met.

This is a developer API for solver packages. Call it from a stepping routine at a point
where `J`, `fu` and `u` all describe the *same* iterate — in practice immediately after
the Jacobian is formed and before a descent is taken from it. Supplying a Jacobian from a
previous iterate would test stationarity at a point the solver has already left.

The criterion is a disjunction with, not a replacement for, the residual test the
termination mode already performs: the residual arm is cheap and it covers the
zero-residual case, where the measure is a ratio of two quantities going to zero and
carries no information.

The verdict depends on the problem type, because the same measurement means different
things:

| | `JᵀF ≈ 0`, `F` small | `JᵀF ≈ 0`, `F` not small |
|---|---|---|
| `NonlinearLeastSquaresProblem` | `ReturnCode.Success` | `ReturnCode.Success` |
| `NonlinearProblem` | `ReturnCode.Success` | `ReturnCode.Stalled` |

For a least-squares problem a stationary point *is* the solution. For a square root-find
it is a local minimum of `‖F‖` that is not a root, which is a failure — and reporting it
as `ReturnCode.Stalled` turns a solve that would otherwise exhaust `maxiters` into a
statement about what went wrong.

# Arguments

  - `cache`: A solver cache carrying `termination_cache`, `retcode` and `force_stop`.
  - `J`: Jacobian at `u`.
  - `fu`: Residual at `u`.
  - `u`: Current iterate.

# Returns

`true` when the solve was stopped, `false` otherwise. It returns `false` without computing
anything when the termination mode has no `gtol`, so a mode that does not opt in — which
is every mode by default, including any user-defined one — pays nothing and behaves
exactly as before.

# Examples

```julia
using NonlinearSolve, NonlinearSolveBase

prob = NonlinearLeastSquaresProblem((u, p) -> [u[1] - 1.0, 2.0], [0.0])
cache = init(prob, LevenbergMarquardt())
NonlinearSolveBase.check_gradient_and_update!(cache, [1.0; 0.0;;], get_fu(cache), get_u(cache))
```
"""
function check_gradient_and_update! end

"""
    default_gradient_tolerance(T) -> Real

Return the default `gtol` for the gradient-stationarity criterion at element type `T`.

The measure of [`gradient_stationarity_measure`](@ref) is a cosine, so the tolerance is
dimensionless and depends only on working precision. `sqrt(eps(T))` sits well above the
`O(eps * sqrt(m))` floor that cancellation in `JᵀF` imposes, so the criterion stays
attainable, while a cosine that small already certifies the residual as orthogonal to
every column of the Jacobian.

# Arguments

  - `T`: Element type of the iterate.

# Returns

The default tolerance as a real number.

# Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.default_gradient_tolerance(Float64)
```
"""
function default_gradient_tolerance end
