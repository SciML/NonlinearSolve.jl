# Faster Startup and and Static Compilation

In many instances one may want a very lightweight version of NonlinearSolve.jl. For this case there
exists the solver package SimpleNonlinearSolve.jl. SimpleNonlinearSolve.jl solvers all satisfy the
same interface as NonlinearSolve.jl, but they are designed to be simpler, lightweight, and thus
have a faster startup time. Everything that can be done with NonlinearSolve.jl can be done with
SimpleNonlinearSolve.jl. Thus for example, we can solve the core tutorial problem with just SimpleNonlinearSolve.jl
as follows:

```@example simple
using SimpleNonlinearSolve

f(u, p) = u .* u .- p
u0 = [1.0, 1.0]
p = 2.0
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, SimpleNewtonRaphson())
```

However, there are a few downsides to SimpleNonlinearSolve's `SimpleX` style algorithms to note:

 1. SimpleNonlinearSolve.jl's methods are not hooked into the LinearSolve.jl system, and thus do not have
    the ability to specify linear solvers, use sparse matrices, preconditioners, and all of the other features
    which are required to scale for very large systems of equations.
 2. SimpleNonlinearSolve.jl's methods have less robust error handling and termination conditions, and thus
    these methods are missing some flexibility and give worse hints for debugging.
 3. SimpleNonlinearSolve.jl's methods are focused on out-of-place support. There is some in-place support,
    but it's designed for simple smaller systems and so some optimizations are missing.

However, the major upsides of SimpleNonlinearSolve.jl are:

 1. The methods are optimized and non-allocating on StaticArrays
 2. The methods are minimal in compilation

As such, you can use the code as shown above to have very low startup with good methods, but for more scaling and debuggability
we recommend the full NonlinearSolve.jl. But that said,

```@example simple
using StaticArrays

u0 = SA[1.0, 1.0]
p = 2.0
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, SimpleNewtonRaphson())
```

using StaticArrays.jl is also the fastest form for small equations, so if you know your system is small then SimpleNonlinearSolve.jl
is not only sufficient but optimal.

## Static Compilation

Julia has tools for building small binaries via static compilation with [StaticCompiler.jl](https://github.com/tshort/StaticCompiler.jl).
However, these tools are currently limited to type-stable non-allocating functions. That said, SimpleNonlinearSolve.jl's solvers are
precisely the subset of NonlinearSolve.jl which are compatible with static compilation.
