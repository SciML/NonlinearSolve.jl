# Descent Subroutines

The following subroutines are available for computing the descent direction.

```@index
Pages = ["descent.md"]
```

## Core Subroutines

```@docs
NewtonDescent
SteepestDescent
DampedNewtonDescent
```

## Special Trust Region Descent Subroutines

```@docs
Dogleg
NonlinearSolveBase.MoreTrustRegionDescent
NonlinearSolveBase.TrustRegionSubproblem
NonlinearSolveBase.TrustRegionScaling
```

## Special Levenberg Marquardt Descent Subroutines

```@docs
GeodesicAcceleration
```
