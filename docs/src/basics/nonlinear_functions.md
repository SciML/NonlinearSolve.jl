# [Nonlinear Functions and Jacobian Types](@id nonlinearfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions
associated with the differential equation's data. In traditional libraries, there is usually
only one option: the Jacobian. However, we allow for a large array of pre-computed functions
to speed up the calculations. This is offered via the `NonlinearFunction` types, which can
be passed to the problems.

## Function Type Definitions

```@docs
SciMLBase.IntervalNonlinearFunction
SciMLBase.NonlinearFunction
```
