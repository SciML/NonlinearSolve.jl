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
NonlinearSolveBase.assert_extension_supported_termination_condition
NonlinearSolveBase.construct_extension_function_wrapper
NonlinearSolveBase.construct_extension_jac
NonlinearSolveBase.select_forward_mode_autodiff
NonlinearSolveBase.select_reverse_mode_autodiff
NonlinearSolveBase.select_jacobian_autodiff
NonlinearSolveBase.L2_NORM
NonlinearSolveBase.Linf_NORM
NonlinearSolveBase.UNITLESS_ABS2
NonlinearSolveBase.NAN_CHECK
NonlinearSolveBase.get_tolerance
NonlinearSolveBase.nonlinearsolve_forwarddiff_solve
NonlinearSolveBase.nonlinearsolve_dual_solution
```
