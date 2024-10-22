# Internal Algorithm Helpers

## Pseudo Transient Method

```@docs
NonlinearSolve.SwitchedEvolutionRelaxation
NonlinearSolve.SwitchedEvolutionRelaxationCache
```

## Approximate Jacobian Methods

### Initialization

```@docs
NonlinearSolve.IdentityInitialization
NonlinearSolve.TrueJacobianInitialization
NonlinearSolve.BroydenLowRankInitialization
```

### Jacobian Structure

```@docs
NonlinearSolve.FullStructure
NonlinearSolve.DiagonalStructure
```

### Jacobian Caches

```@docs
NonlinearSolve.InitializedApproximateJacobianCache
```

### Reset Methods

```@docs
NonlinearSolve.NoChangeInStateReset
NonlinearSolve.IllConditionedJacobianReset
```

### Update Rules

```@docs
NonlinearSolve.GoodBroydenUpdateRule
NonlinearSolve.BadBroydenUpdateRule
NonlinearSolve.KlementUpdateRule
```

## Levenberg Marquardt Method

```@docs
NonlinearSolve.LevenbergMarquardtTrustRegion
```

## Trust Region Method

```@docs
NonlinearSolve.GenericTrustRegionScheme
```

## Miscellaneous

```@docs
NonlinearSolve.callback_into_cache!
NonlinearSolve.concrete_jac
```
