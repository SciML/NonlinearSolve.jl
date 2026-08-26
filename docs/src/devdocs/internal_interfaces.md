# Internal Abstract Types

This section documents developer public API used by NonlinearSolve.jl subpackages and
downstream solver implementations. These names are versioned extension points, but they are
not the recommended user-facing API for solving nonlinear systems.

## Developer API Namespace

```@docs
NonlinearSolveBase
NonlinearSolveBase.InternalAPI
```

## Problem Concretization

```@docs
NonlinearSolveBase.get_concrete_problem
```

## SCC Interface

```@docs
SCCNonlinearSolve.scc_solve_up
```

## Solvers

```@docs
NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
NonlinearSolveBase.AbstractNonlinearSolveCache
NonlinearSolveBase.NonlinearSolveNoInitCache
```

## Nonlinear Preconditioning

Hooks backing the `precondition` and `postcondition` solve options described in
[Nonlinear Preconditioning](@ref nonlinear_preconditioning). `transform_conditioned_problem`
runs at the `solve`/`init` funnels, while `apply_postcondition!!` is called by each solver
family at its iterate-commit points.

```@docs
NonlinearSolveBase.get_precondition
NonlinearSolveBase.get_postcondition
NonlinearSolveBase.needs_conditioning
NonlinearSolveBase.transform_conditioned_problem
NonlinearSolveBase.apply_postcondition!!
NonlinearSolveBase.supports_postcondition
```

## Descent Directions

```@docs
NonlinearSolveBase.AbstractDescentDirection
NonlinearSolveBase.AbstractDescentCache
NonlinearSolveBase.supports_line_search
NonlinearSolveBase.supports_trust_region
NonlinearSolveBase.set_du!
NonlinearSolveBase.last_step_accepted
NonlinearSolveBase.preinverted_jacobian
NonlinearSolveBase.normal_form
```

### Descent Results

```@docs
NonlinearSolveBase.DescentResult
```

## Approximate Jacobian

```@docs
NonlinearSolveBase.AbstractApproximateJacobianStructure
NonlinearSolveBase.AbstractJacobianInitialization
NonlinearSolveBase.AbstractApproximateJacobianUpdateRule
NonlinearSolveBase.AbstractApproximateJacobianUpdateRuleCache
NonlinearSolveBase.reset_update_rule_state!
NonlinearSolveBase.AbstractResetCondition
NonlinearSolveBase.stores_full_jacobian
NonlinearSolveBase.get_full_jacobian
NonlinearSolveBase.jacobian_initialized_preinverted
NonlinearSolveBase.store_inverse_jacobian
```

## Damping Algorithms

```@docs
NonlinearSolveBase.AbstractDampingFunction
NonlinearSolveBase.AbstractDampingFunctionCache
NonlinearSolveBase.requires_normal_form_jacobian
NonlinearSolveBase.requires_normal_form_rhs
NonlinearSolveBase.returns_norm_form_damping
```

## Trust Region

```@docs
NonlinearSolveBase.AbstractTrustRegionMethod
NonlinearSolveBase.AbstractTrustRegionMethodCache
```

## Cache State

Accessors for the state of a running solve. These are the only cache accessors a
`postcondition` corrector should use on the cache it is handed.

```@docs
NonlinearSolveBase.get_u
NonlinearSolveBase.get_fu
NonlinearSolveBase.get_nsteps
```

## Cache Drivers

The stepping driver is used for iterative caches. `solve_cache!` is the allocation-sensitive
completion path; it is not available for `NonlinearSolveNoInitCache`.

The public stepping entry point is `CommonSolve.step!`, while the
allocation-sensitive completion entry point is `NonlinearSolveBase.solve_cache!`.

```@docs
NonlinearSolveBase.get_termination_cache
NonlinearSolveBase.get_trace
NonlinearSolveBase.solve_cache!
```

## Deferred Residual Evaluation

A solver whose step ends by evaluating the residual at the iterate it just produced spends
that evaluation on the *next* step's right-hand side. A driver that stops stepping — an
implicit ODE integrator taking one Newton iteration per outer iteration, say — throws the
last one away. These two let it ask for that evaluation to be skipped and take it later only
if it turns out to want it.

```@docs
NonlinearSolveBase.NonlinearSolveTrace
NonlinearSolveBase.supports_deferred_residual
NonlinearSolveBase.refresh_residual!
NonlinearSolveBase.residual_only_termination_mode
NonlinearSolveBase.trace_is_active
```

## Cache Tolerances

```@docs
NonlinearSolveBase.get_abstol
NonlinearSolveBase.get_reltol
```

## Termination Mode Supertypes

```@docs
NonlinearSolveBase.AbstractNonlinearTerminationMode
NonlinearSolveBase.AbstractSafeNonlinearTerminationMode
```
