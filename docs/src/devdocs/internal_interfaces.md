# Internal Abstract Types

This section documents developer public API used by NonlinearSolve.jl subpackages and
downstream solver implementations. These names are versioned extension points, but they are
not the recommended user-facing API for solving nonlinear systems.

## Developer API Namespace

```@docs
NonlinearSolveBase.InternalAPI
```

## Solvers

```@docs
NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
NonlinearSolveBase.AbstractNonlinearSolveCache
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
