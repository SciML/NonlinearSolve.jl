# [NonlinearFunctions and Jacobian Types](@id nonlinearfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions
associated with the differential equation's data. In traditional libraries there
is usually only one option: the Jacobian. However, we allow for a large array
of pre-computed functions to speed up the calculations. This is offered via the
`NonlinearFunction` types, which can be passed to the problems.

## Function Type Definitions

### Function Choice Definitions

The full interface available to the solvers is as follows:

- `jac`: The Jacobian of the differential equation with respect to the state
  variable `u` at a time `t` with parameters `p`.
- `tgrad`: The gradient of the differential equation with respect to `t` at state
  `u` with parameters `p`.
- `paramjac`: The Jacobian of the differential equation with respect to `p` at
  state `u` at time `t`.
- `analytic`: Defines an analytical solution using `u0` at time `t` with `p`
  which will cause the solvers to return errors. Used for testing.
- `syms`: Allows you to name your variables for automatic names in plots and
  other output.

### NonlinearFunction

```julia
function NonlinearFunction{iip,true}(f;
  analytic=nothing, # (u0,p)
  jac=nothing, # (J,u,p) or (u,p)
  jvp=nothing,
  vjp=nothing,
  jac_prototype=nothing, # Type for the Jacobian
  sparsity=jac_prototype,
  paramjac = nothing,
  syms = nothing,
  observed = DEFAULT_OBSERVED_NO_TIME,
  colorvec = nothing)
```
