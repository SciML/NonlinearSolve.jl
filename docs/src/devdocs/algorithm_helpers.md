# Internal Algorithm Helpers

## Pseudo Transient Method

```@docs
NonlinearSolveFirstOrder.SwitchedEvolutionRelaxation
NonlinearSolveFirstOrder.SwitchedEvolutionRelaxationCache
```

## Approximate Jacobian Methods

### Initialization

```@docs
NonlinearSolveQuasiNewton.IdentityInitialization
NonlinearSolveQuasiNewton.TrueJacobianInitialization
NonlinearSolveQuasiNewton.BroydenLowRankInitialization
```

### Jacobian Structure

```@docs
NonlinearSolveQuasiNewton.FullStructure
NonlinearSolveQuasiNewton.DiagonalStructure
```

### Jacobian Caches

```@docs
NonlinearSolveQuasiNewton.InitializedApproximateJacobianCache
```

### Reset Methods

```@docs
NonlinearSolveQuasiNewton.NoChangeInStateReset
NonlinearSolveQuasiNewton.IllConditionedJacobianReset
```

### Update Rules

```@docs
NonlinearSolveQuasiNewton.GoodBroydenUpdateRule
NonlinearSolveQuasiNewton.BadBroydenUpdateRule
NonlinearSolveQuasiNewton.KlementUpdateRule
```

## Levenberg Marquardt Method

```@docs
NonlinearSolveFirstOrder.LevenbergMarquardtTrustRegion
```

## Trust Region Method

```@docs
NonlinearSolveFirstOrder.GenericTrustRegionScheme
```

## Miscellaneous

```@docs
NonlinearSolveBase.callback_into_cache!
NonlinearSolveBase.concrete_jac
```
