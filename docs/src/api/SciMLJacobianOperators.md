```@meta
CurrentModule = SciMLJacobianOperators
```

# SciMLJacobianOperators.jl

This is a subpackage on NonlinearSolve providing a general purpose JacVec and VecJac
operator built on top on DifferentiationInterface.jl.

```julia
import Pkg
Pkg.add("SciMLJacobianOperators")
import SciMLJacobianOperators
```

## Jacobian API

```@docs
JacobianOperator
VecJacOperator
JacVecOperator
```

## Stateful Operators

```@docs
StatefulJacobianOperator
StatefulJacobianNormalFormOperator
```
