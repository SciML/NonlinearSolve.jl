# [Code Optimization for Solving Nonlinear Systems](@id code_optimization)

## Optimizing Nonlinear Solver Code for Small Systems

```@example
using NonlinearSolve, StaticArrays

f(u, p) = u .* u .- p
u0 = @SVector[1.0, 1.0]
p = 2.0
probN = NonlinearProblem(f, u0, p)
sol = solve(probN, NewtonRaphson(), reltol = 1e-9)
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
prob = NonlinearProblem(f!, u0, p)

linsolve = LinearSolve.KrylovJL_GMRES()
sol = solve(prob, NewtonRaphson(; linsolve), reltol = 1e-9)
```