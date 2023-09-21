# Solving Nonlinear Systems

A nonlinear system $$f(u) = 0$$ is specified by defining a function `f(u,p)`,
where `p` are the parameters of the system. For example, the following solves
the vector equation $$f(u) = u^2 - p$$ for a vector of equations:

```@example
using NonlinearSolve, StaticArrays

f(u, p) = u .* u .- p
u0 = @SVector[1.0, 1.0]
p = 2.0
probN = NonlinearProblem(f, u0, p)
sol = solve(probN, NewtonRaphson(), reltol = 1e-9)
```

where `u0` is the initial condition for the rootfinder. Native NonlinearSolve.jl
solvers use the given type of `u0` to determine the type used within the solver
and the return. Note that the parameters `p` can be any type, but most are an
AbstractArray for automatic differentiation.

## Using Bracketing Methods

For scalar rootfinding problems, bracketing methods exist in `SimpleNonlinearSolve`. In this case, one passes
a bracket instead of an initial condition, for example:

```@example
using SimpleNonlinearSolve
f(u, p) = u * u - 2.0
uspan = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, uspan)
sol = solve(probB, ITP())
```

The user can also set a tolerance that suits the application.

```@example
using SimpleNonlinearSolve
f(u, p) = u * u - 2.0
uspan = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, uspan)
sol = solve(probB, ITP(), abstol = 0.01)
```

## Using Jacobian Free Newton Krylov (JNFK) Methods

If we want to solve the first example, without constructing the entire Jacobian

```@example
using NonlinearSolve, LinearSolve

function f!(res, u, p)
    @. res = u * u - p
end
u0 = [1.0, 1.0]
p = 2.0
probN = NonlinearProblem(f!, u0, p)

linsolve = LinearSolve.KrylovJL_GMRES()
sol = solve(probN, NewtonRaphson(; linsolve), reltol = 1e-9)
```
