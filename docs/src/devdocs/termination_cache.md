# Termination Cache API

This API is for solver-package developers that evaluate a nonlinear termination condition
outside NonlinearSolve's standard solve loop. Application code should use `solve` and
configured termination modes rather than constructing or inspecting termination caches.

```@docs
NonlinearSolveBase.termination_condition_result
```
