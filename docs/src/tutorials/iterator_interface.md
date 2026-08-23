# [Nonlinear Solver Iterator Interface](@id iterator)

There is an iterator form of the nonlinear solver which somewhat mirrors the DiffEq
integrator interface:

```@example iterator_interface
import NonlinearSolve as NLS
import NonlinearSolveBase as NLSB

f(u, p) = u .* u .- 2.0
u0 = 1.5
probB = NLS.NonlinearProblem(f, u0)

nlcache = NLS.init(probB, NLS.NewtonRaphson())
```

`init` takes the same keyword arguments as [`solve`](@ref solver_options), but it returns a
cache object that satisfies `typeof(nlcache) <: AbstractNonlinearSolveCache`. There are two
cache forms:

  - Native iterative algorithms return a stepping cache. Call `step!` to advance it, or
    `solve!` to run it to completion.
  - Algorithms without a `SciMLBase.__init` method, such as the `SimpleNonlinearSolve`
    algorithms, return [`NonlinearSolveNoInitCache`](@ref NonlinearSolveBase.NonlinearSolveNoInitCache).
    This cache stores the problem and
    options but no iteration state, so call `solve!` directly; `step!`, `get_fu`, and
    `get_nsteps` are unavailable.

The iterator interface supports:

```@docs
step!(nlcache::NonlinearSolveBase.AbstractNonlinearSolveCache, args...; kwargs...)
```

We can perform 10 steps of the Newton-Raphson solver with the following:

```@example iterator_interface
for i in 1:10
    NLS.step!(nlcache)
end
```

Code that accepts any nonlinear algorithm can detect the second form and choose the complete
solve path. The stepping branch can use `solve_cache!` when it needs the allocation-sensitive
cache result:

```@example iterator_interface
function solve_any(prob, alg)
    cache = NLS.init(prob, alg)
    if cache isa NLSB.NonlinearSolveNoInitCache
        return NLS.solve!(cache)
    end
    NLSB.solve_cache!(cache)
    return NLS.solve!(cache)
end

simple_sol = solve_any(probB, NLS.SimpleNewtonRaphson())
```

We currently don't implement a `Base.iterate` interface but that will be added in the
future.
