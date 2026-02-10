# [Nonlinear Problems](@id problems)

## The Four Types of Nonlinear Problems

NonlinearSolve.jl tackles four related types of nonlinear systems:

 1. Interval rootfinding problems. I.e., find the ``t \in [t_0, t_f]`` such that
    ``f(t) = 0``.
 2. Systems of nonlinear equations, i.e., find the ``u`` such that ``f(u) = 0``.
 3. Steady state problems, i.e., find the ``u`` such that ``u' = f(u,t)`` has reached steady
    state, i.e., ``0 = f(u, ∞)``.
 4. The nonlinear least squares problem, which is an under/over-constrained nonlinear system
    which might not be satisfiable, i.e. there may be no ``u`` such that ``f(u) = 0``, and thus
    we find the ``u`` which minimizes ``\|f(u)\|_2^2`` in the least squares sense.

The first is for solving scalar rootfinding problems, i.e., finding a single number, and
requires that a bracketing interval is known. For a bracketing interval, one must have that
the sign of ``f(t_0)`` is opposite the sign of ``f(t_f)``, thus guaranteeing a root in the
interval.

!!! note
    Interval rootfinding problems allow for ``f`` to return an array, in which case the
    interval rootfinding problem is interpreted as finding the first ``t`` such that any of
    the components of the array hit zero.

The second type of nonlinear system can be multidimensional, and thus no ordering nor
boundaries are assumed to be known. For a system of nonlinear equations, ``f`` can return
an array, and the solver seeks the value of ``u`` for which all outputs of ``f`` are
simultaneously zero.

The third type is equivalent to a nonlinear system, but with the extra interpretation of
having a potentially preferred unique root. That is, when there are multiple `u` such
that ``f(u) = 0``, the `NonlinearProblem` does not have a preferred solution, while for the
`SteadyStateProblem` the preferred solution is the ``u(∞)`` that would arise from solving the
ODE ``u' = f(u,t)``.

The fourth type is an overdetermined nonlinear system, which has more constraints than free
variables, and thus is usually not possible to solve exactly. In these contexts, it is usually
convenient to minimize the Euclidean norm, as it is continuously differentiable if the
original function is.

!!! warning

    Most solvers for `SteadyStateProblem` do not guarantee the preferred solution and
    instead will solve for some `u` in the set of solutions. The documentation of the
    nonlinear solvers will note if they return the preferred solution.

## Problem Construction Details

```@docs
IntervalNonlinearProblem
NonlinearProblem
SteadyStateProblem
NonlinearLeastSquaresProblem
```
