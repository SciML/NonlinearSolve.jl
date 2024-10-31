# Internal Abstract Types

## Solvers

```@docs
NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
NonlinearSolveBase.AbstractNonlinearSolveCache
```

## Descent Directions

```@docs
NonlinearSolveBase.AbstractDescentDirection
NonlinearSolveBase.AbstractDescentCache
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
