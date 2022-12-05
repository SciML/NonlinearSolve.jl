# Nonlinear Problems

## The Three Types of Nonlinear Problems

NonlinearSolve.jl tackles three related types of nonlinear systems:

1. Interval rootfinding problems. I.e., find the ``t \in [t_0, t_f]`` such that ``f(t) = 0``.
2. Systems of nonlinear equations, i.e. find the ``u`` such that ``f(u) = 0``.
3. Steady state problems, i.e. find the ``u`` such that ``u' = f(u,t)`` has reached steady state,
   i.e. ``0 = f(u, ∞)``.

The first is for solving scalar rootfinding problems, i.e. finding a single number, and
requires that a bracketing interval is known. For a bracketing interval, one must have that
the sign of `f(t_0)` is opposite the sign of `f(t_f)`, thus guaranteeing a root in the
interval.

!!! note

    Interval rootfinding problems allow for `f` to return an array, in which case the interval
    rootfinding problem is interpreted as finding the first `t` such that any of the components
    of the array hit zero.

The second type of nonlinear system can be multidimensional and thus no ordering nor
boundaries are assumed to be known. For a system of nonlinear equations, `f` can return
an array and the solver seeks to find the value of `u` for which all outputs of `f` are
simultaniously zero.

The last type if equivalent to a nonlinear system but with the extra interpretion of
having a potentially preferred unique root. That is, when there are multiple `u` such
that `f(u) = 0`, the `NonlinearProblem` does not have a preferred solution, while for the
`SteadyStateProblem` the preferred solution is the `u(∞)` that would arise from solving the
ODE `u' = f(u,t)`.

!!! warn

    Most solvers for `SteadyStateProblem` do not guarentee the preferred solution and
    instead will solve for some `u` in the set of solutions. The documentation of the
    nonlinear solvers will note if they return the preferred solution.

## Problem Construction Details

```@docs
SciMLBase.IntervalNonlinearProblem
SciMLBase.NonlinearProblem
SciMLBase.SteadyStateProblem
```
