# [Nonlinear Solver Iterator Interface](@id iterator)

There is an iterator form of the nonlinear solver which somewhat mirrors the DiffEq
integrator interface:

```@example iterator_interface
using NonlinearSolve

f(u, p) = u .* u .- 2.0
u0 = 1.5
probB = NonlinearProblem(f, u0)

nlcache = init(probB, NewtonRaphson())
```

`init` takes the same keyword arguments as [`solve`](@ref solver_options), but it returns a
cache object that satisfies `typeof(nlcache) <: AbstractNonlinearSolveCache` and can be used
to iterate the solver.

The iterator interface supports:

```@docs
step!(nlcache::NonlinearSolveBase.AbstractNonlinearSolveCache, args...; kwargs...)
```

We can perform 10 steps of the Newton-Raphson solver with the following:

```@example iterator_interface
for i in 1:10
    step!(nlcache)
end
```

We currently don't implement a `Base.iterate` interface but that will be added in the
future.
