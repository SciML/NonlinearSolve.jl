# Nonlinear Problems

## The Two Types of Nonlinear Problems

NonlinearSolve.jl tackles two related types of nonlinear systems:

1. Interval rootfinding problems. I.e., find the ``t in [t_0, t_f]`` such that `f(t) = 0`.
2. Systems of nonlinear equations, i.e. find the `u` such that `f(u) = 0`.

The former is for solving scalar rootfinding problems, i.e. finding a single number, and
requires that a bracketing interval is known. For a bracketing interval, one must have that
the sign of `f(t_0)` is opposite the sign of `f(t_f)`, thus guaranteeing a root in the
interval.

The latter type of nonlinear system can be multidimensional and thus no ordering nor
boundaries are assumed to be known. For a system of nonlinear equations, `f` can return
an array and the solver seeks to find the value of `u` for which all outputs of `f` are
simultaniously zero.

!!! note

    Interval rootfinding problems allow for `f` to return an array, in which case the interval
    rootfinding problem is interpreted as finding the first `t` such that any of the components
    of the array hit zero.


## Problem Construction Details

```@docs
SciMLBase.IntervalNonlinearProblem
SciMLBase.NonlinearProblem
```
