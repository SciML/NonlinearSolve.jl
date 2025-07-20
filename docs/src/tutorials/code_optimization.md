# [Code Optimization for Small Nonlinear Systems in Julia](@id code_optimization)

## General Code Optimization in Julia

Before starting this tutorial, we recommend the reader to check out one of the many
tutorials for optimization Julia code. The following is an incomplete list:

  - [The Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
  - [MIT 18.337 Course Notes on Optimizing Serial Code](https://mitmath.github.io/18337/lecture2/optimizing)
  - [What scientists must know about hardware to write fast code](https://viralinstruction.com/posts/hardware/)

User-side optimizations are important because, for sufficiently difficult problems, most
time will be spent inside your `f` function, the function you are trying to solve.
“Efficient solvers" are those that reduce the required number of `f` calls to hit the error
tolerance. The main ideas for optimizing your nonlinear solver code, or any Julia function,
are the following:

  - Make it non-allocating
  - Use StaticArrays for small arrays
  - Use broadcast fusion
  - Make it type-stable
  - Reduce redundant calculations
  - Make use of BLAS calls
  - Optimize algorithm choice

We'll discuss these strategies in the context of nonlinear solvers. Let's start with small
systems.

## Optimizing Nonlinear Solver Code for Small Systems

Take for example a prototypical small nonlinear solver code in its out-of-place form:

```@example small_opt
import NonlinearSolve as NLS

f(u, p) = u .* u .- p
u0 = [1.0, 1.0]
p = 2.0
prob = NLS.NonlinearProblem(f, u0, p)
sol = NLS.solve(prob, NLS.NewtonRaphson())
```

We can use BenchmarkTools.jl to get more precise timings:

```@example small_opt
import BenchmarkTools

BenchmarkTools.@benchmark NLS.solve(prob, NLS.NewtonRaphson())
```

Note that this way of writing the function is a shorthand for:

```@example small_opt
f(u, p) = [u[1] * u[1] - p, u[2] * u[2] - p]
```

where the function `f` returns an array. This is a common pattern from things like MATLAB's
`fzero` or SciPy's `scipy.optimize.fsolve`. However, by design it's very slow. In the
benchmark you can see that there are many allocations. These allocations cause the memory
allocator to take more time than the actual numerics itself, which is one of the reasons why
codes from MATLAB and Python end up slow.

In order to avoid this issue, we can use a non-allocating "in-place" approach. Written out
by hand, this looks like:

```@example small_opt
function f(du, u, p)
    du[1] = u[1] * u[1] - p
    du[2] = u[2] * u[2] - p
    return nothing
end

prob = NLS.NonlinearProblem(f, u0, p)
BenchmarkTools.@benchmark sol = NLS.solve(prob, NLS.NewtonRaphson())
```

Notice how much faster this already runs! We can make this code even simpler by using
the `.=` in-place broadcasting.

```@example small_opt
function f(du, u, p)
    du .= u .* u .- p
    return nothing
end

BenchmarkTools.@benchmark sol = NLS.solve(prob, NLS.NewtonRaphson())
```

## Further Optimizations for Small Nonlinear Solves with Static Arrays and SimpleNonlinearSolve

Allocations are only expensive if they are “heap allocations”. For a more in-depth
definition of heap allocations,
[there are many sources online](https://net-informations.com/faq/net/stack-heap.htm).
But a good working definition is that heap allocations are variable-sized slabs of memory
which have to be pointed to, and this pointer indirection costs time. Additionally, the heap
has to be managed, and the garbage controllers has to actively keep track of what's on the
heap.

However, there's an alternative to heap allocations, known as stack allocations. The stack
is statically-sized (known at compile time) and thus its accesses are quick. Additionally,
the exact block of memory is known in advance by the compiler, and thus re-using the memory
is cheap. This means that allocating on the stack has essentially no cost!

Arrays have to be heap allocated because their size (and thus the amount of memory they take
up) is determined at runtime. But there are structures in Julia which are stack-allocated.
`struct`s for example are stack-allocated “value-type”s. `Tuple`s are a stack-allocated
collection. The most useful data structure for NonlinearSolve though is the `StaticArray`
from the package [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl). These
arrays have their length determined at compile-time. They are created using macros attached
to normal array expressions, for example:

```@example small_opt
import StaticArrays

A = StaticArrays.SA[2.0, 3.0, 5.0]
typeof(A)
```

Notice that the `3` after `SVector` gives the size of the `SVector`. It cannot be changed.
Additionally, `SVector`s are immutable, so we have to create a new `SVector` to change
values. But remember, we don't have to worry about allocations because this data structure
is stack-allocated. `SArray`s have numerous extra optimizations as well: they have fast
matrix multiplication, fast QR factorizations, etc. which directly make use of the
information about the size of the array. Thus, when possible, they should be used.

Unfortunately, static arrays can only be used for sufficiently small arrays. After a certain
size, they are forced to heap allocate after some instructions and their compile time
balloons. Thus, static arrays shouldn't be used if your system has more than ~20 variables.
Additionally, only the native Julia algorithms can fully utilize static arrays.

Let's ***optimize our nonlinear solve using static arrays***. Note that in this case, we
want to use the out-of-place allocating form, but this time we want to output a static
array. Doing it with broadcasting looks like:

```@example small_opt
f_SA(u, p) = u .* u .- p

u0 = StaticArrays.SA[1.0, 1.0]
p = 2.0
prob = NLS.NonlinearProblem(f_SA, u0, p)

BenchmarkTools.@benchmark NLS.solve(prob, NLS.NewtonRaphson())
```

Note that only change here is that `u0` is made into a StaticArray! If we needed to write
`f` out for a more complex nonlinear case, then we'd simply do the following:

```@example small_opt
f_SA(u, p) = StaticArrays.SA[u[1] * u[1] - p, u[2] * u[2] - p]

BenchmarkTools.@benchmark NLS.solve(prob, NLS.NewtonRaphson())
```

However, notice that this did not give us a speedup but rather a slowdown. This is because
many of the methods in NonlinearSolve.jl are designed to scale to larger problems, and thus
aggressively do things like caching which is good for large problems but not good for these
smaller problems and static arrays. In order to see the full benefit, we need to move to one
of the methods from SimpleNonlinearSolve.jl which are designed for these small-scale static
examples. Let's now use `SimpleNewtonRaphson`:

```@example small_opt
BenchmarkTools.@benchmark NLS.solve(prob, NLS.SimpleNewtonRaphson())
```

And there we go, around `40ns` from our starting point of almost `4μs`!
