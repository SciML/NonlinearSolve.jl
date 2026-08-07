# Internal Abstract Types

This section documents developer public API used by NonlinearSolve.jl subpackages and
downstream solver implementations. These names are versioned extension points, but they are
not the recommended user-facing API for solving nonlinear systems.

## Developer API Namespace

```@docs
NonlinearSolveBase
NonlinearSolveBase.InternalAPI
```

## SCC Interface

```@docs
SCCNonlinearSolve.scc_solve_up
```

## Solvers

```@docs
NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
NonlinearSolveBase.AbstractNonlinearSolveCache
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
```

## Damping Algorithms

```@docs
NonlinearSolveBase.AbstractDampingFunction
NonlinearSolveBase.AbstractDampingFunctionCache
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
