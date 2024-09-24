```@meta
CurrentModule = SciMLJacobianOperators
```

# SciMLJacobianOperators.jl

This is a subpackage on NonlinearSolve providing a general purpose JacVec and VecJac
operator built on top on DifferentiationInterface.jl.

```julia
import Pkg
Pkg.add("SciMLJacobianOperators")
using SciMLJacobianOperators
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
