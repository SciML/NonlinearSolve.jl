# Solving Nonlinear Systems

A nonlinear system $$f(u) = 0$$ is specified by defining a function `f(u,p)`,
where `p` are the parameters of the system. For example, the following solves
the vector equation $$f(u) = u^2 - p$$ for a vector of equations:

```julia
using NonlinearSolve, StaticArrays

f(u,p) = u .* u .- p
u0 = @SVector[1.0, 1.0]
p = 2.0
probN = NonlinearProblem{false}(f, u0, p)
solver = solve(probN, NewtonRaphson(), tol = 1e-9)
```

where `u0` is the initial condition for the rootfind. Native NonlinearSolve.jl
solvers use the given type of `u0` to determine the type used within the solver
and the return. Note that the parameters `p` can be any type, but most are an
AbstractArray for automatic differentiation.

## Using Bracketing Methods

For scalar rootfinding problems, bracketing methods exist. In this case, one passes
a bracket instead of an initial condition, for example:

```julia
f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = NonlinearProblem(f, u0)
sol = solve(probB, Falsi())
```
