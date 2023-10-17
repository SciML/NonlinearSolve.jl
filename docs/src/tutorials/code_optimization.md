# [Code Optimization for Solving Nonlinear Systems](@id code_optimization)

## Code Optimization in Julia

Before starting this tutorial, we recommend the reader to check out one of the
many tutorials for optimization Julia code. The following is an incomplete
list:

  - [The Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
  - [MIT 18.337 Course Notes on Optimizing Serial Code](https://mitmath.github.io/18337/lecture2/optimizing)
  - [What scientists must know about hardware to write fast code](https://viralinstruction.com/posts/hardware/)

User-side optimizations are important because, for sufficiently difficult problems,
most time will be spent inside your `f` function, the function you are
trying to solve. “Efficient solvers" are those that reduce the required
number of `f` calls to hit the error tolerance. The main ideas for optimizing
your nonlinear solver code, or any Julia function, are the following:

  - Make it non-allocating
  - Use StaticArrays for small arrays
  - Use broadcast fusion
  - Make it type-stable
  - Reduce redundant calculations
  - Make use of BLAS calls
  - Optimize algorithm choice

We'll discuss these strategies in the context of nonlinear solvers.
Let's start with small systems.

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