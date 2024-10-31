# [Efficiently Solving Large Sparse Ill-Conditioned Nonlinear Systems in Julia](@id large_systems)

This tutorial is for getting into the extra features of using NonlinearSolve.jl. Solving
ill-conditioned nonlinear systems requires specializing the linear solver on properties of
the Jacobian in order to cut down on the ``\mathcal{O}(n^3)`` linear solve and the
``\mathcal{O}(n^2)`` back-solves. This tutorial is designed to explain the advanced usage of
NonlinearSolve.jl by solving the steady state stiff Brusselator partial differential
equation (BRUSS) using NonlinearSolve.jl.

## Definition of the Brusselator Equation

!!! note
    
    Feel free to skip this section: it simply defines the example problem.

The Brusselator PDE is defined as follows:

```math
\begin{align}
0 &= 1 + u^2v - 4.4u + \alpha(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t)\\
0 &= 3.4u - u^2v + \alpha(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2})
\end{align}
```

where

```math
f(x, y, t) = \begin{cases}
5 & \quad \text{if } (x-0.3)^2+(y-0.6)^2 ≤ 0.1^2 \text{ and } t ≥ 1.1 \\
0 & \quad \text{else}
\end{cases}
```

and the initial conditions are

```math
\begin{align}
u(x, y, 0) &= 22\cdot (y(1-y))^{3/2} \\
v(x, y, 0) &= 27\cdot (x(1-x))^{3/2}
\end{align}
```

with the periodic boundary condition

```math
\begin{align}
u(x+1,y,t) &= u(x,y,t) \\
u(x,y+1,t) &= u(x,y,t)
\end{align}
```

To solve this PDE, we will discretize it into a system of ODEs with the finite difference
method. We discretize `u` and `v` into arrays of the values at each time point:
`u[i,j] = u(i*dx,j*dy)` for some choice of `dx`/`dy`, and same for `v`. Then our ODE is
defined with `U[i,j,k] = [u v]`. The second derivative operator, the Laplacian, discretizes
to become a tridiagonal matrix with `[1 -2 1]` and a `1` in the top right and bottom left
corners. The nonlinear functions are then applied at each point in space (they are
broadcast). Use `dx=dy=1/32`.

The resulting `NonlinearProblem` definition is:

```@example ill_conditioned_nlprob
using NonlinearSolve, LinearAlgebra, SparseArrays, LinearSolve

const N = 32
const xyd_brusselator = range(0, stop = 1, length = N)

brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function brusselator_2d_loop(du, u, p)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        du[i, j, 1] = alpha * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                       4u[i, j, 1]) +
                      B +
                      u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y)
        du[i, j, 2] = alpha * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                       4u[i, j, 2]) + A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end
p = (3.4, 1.0, 10.0, step(xyd_brusselator))

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end

u0 = init_brusselator_2d(xyd_brusselator)
prob_brusselator_2d = NonlinearProblem(
    brusselator_2d_loop, u0, p; abstol = 1e-10, reltol = 1e-10
)
```

## Choosing Jacobian Types

When we are solving this nonlinear problem, the Jacobian must be built at many iterations,
and this can be one of the most expensive steps. There are two pieces that must be optimized
in order to reach maximal efficiency when solving stiff equations: the sparsity pattern and
the construction of the Jacobian. The construction is filling the matrix `J` with values,
while the sparsity pattern is what `J` to use.

The sparsity pattern is given by a prototype matrix, the `jac_prototype`, which will be
copied to be used as `J`. The default is for `J` to be a `Matrix`, i.e. a dense matrix.
However, if you know the sparsity of your problem, then you can pass a different matrix
type. For example, a `SparseMatrixCSC` will give a sparse matrix. Other sparse matrix types
include:

  - Bidiagonal
  - Tridiagonal
  - SymTridiagonal
  - BandedMatrix ([BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl))
  - BlockBandedMatrix ([BlockBandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BlockBandedMatrices.jl))

## Approximate Sparsity Detection & Sparse Jacobians

In the next section, we will show how to specify `sparsity` to trigger automatic sparsity
detection.

```@example ill_conditioned_nlprob
using BenchmarkTools # for @btime

@btime solve(prob_brusselator_2d, NewtonRaphson());
nothing # hide
```

```@example ill_conditioned_nlprob
using SparseConnectivityTracer

prob_brusselator_2d_autosparse = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop; sparsity = TracerSparsityDetector()),
    u0, p; abstol = 1e-10, reltol = 1e-10
)

@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12)));
@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12),
        linsolve = KLUFactorization()));
@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12),
        linsolve = KrylovJL_GMRES()));
nothing # hide
```

## Declaring a Sparse Jacobian with Automatic Sparsity Detection

Jacobian sparsity is declared by the `jac_prototype` argument in the `NonlinearFunction`.
Note that you should only do this if the sparsity is high, for example, 0.1% of the matrix
is non-zeros, otherwise the overhead of sparse matrices can be higher than the gains from
sparse differentiation!

One of the useful companion tools for NonlinearSolve.jl is
[ADTypes.jl](https://github.com/SciML/ADTypes.jl) that specifies the interface for sparsity
detection via [`jacobian_sparsity`](@extref ADTypes.jacobian_sparsity). This allows for
automatic declaration of Jacobian sparsity types. To see this in action, we can give an
example `du` and `u` and call `jacobian_sparsity` on our function with the example
arguments, and it will kick out a sparse matrix with our pattern, that we can turn into our
`jac_prototype`.

!!! tip
    
    External packages like `SparseConnectivityTracer.jl` and `Symbolics.jl` provide the
    actual implementation of sparsity detection.

```@example ill_conditioned_nlprob
using SparseConnectivityTracer, ADTypes

f! = (du, u) -> brusselator_2d_loop(du, u, p)
du0 = similar(u0)
jac_sparsity = ADTypes.jacobian_sparsity(f!, du0, u0, TracerSparsityDetector())
```

Notice that Julia gives a nice print out of the sparsity pattern. That's neat, and would be
tedious to build by hand! Now we just pass it to the `NonlinearFunction` like as before:

```@example ill_conditioned_nlprob
ff = NonlinearFunction(brusselator_2d_loop; jac_prototype = jac_sparsity)
```

Build the `NonlinearProblem`:

```@example ill_conditioned_nlprob
prob_brusselator_2d_sparse = NonlinearProblem(ff, u0, p; abstol = 1e-10, reltol = 1e-10)
```

Now let's see how the version with sparsity compares to the version without:

```@example ill_conditioned_nlprob
@btime solve(prob_brusselator_2d, NewtonRaphson());
@btime solve(prob_brusselator_2d_sparse, NewtonRaphson());
@btime solve(prob_brusselator_2d_sparse, NewtonRaphson(linsolve = KLUFactorization()));
nothing # hide
```

Note that depending on the properties of the sparsity pattern, one may want to try
alternative linear solvers such as `NewtonRaphson(linsolve = KLUFactorization())`
or `NewtonRaphson(linsolve = UMFPACKFactorization())`

## Using Jacobian-Free Newton-Krylov

A completely different way to optimize the linear solvers for large sparse matrices is to
use a Krylov subspace method. This requires choosing a linear solver for changing to a
Krylov method. To swap the linear solver out, we use the `linsolve` command and choose the
GMRES linear solver.

```@example ill_conditioned_nlprob
@btime solve(prob_brusselator_2d, NewtonRaphson(linsolve = KrylovJL_GMRES()));
nothing # hide
```

Notice that this acceleration does not require the definition of a sparsity pattern, and can
thus be an easier way to scale for large problems. For more information on linear solver
choices, see the
[linear solver documentation](https://docs.sciml.ai/DiffEqDocs/stable/features/linear_nonlinear/#linear_nonlinear).
`linsolve` choices are any valid [LinearSolve.jl](https://linearsolve.sciml.ai/dev/) solver.

!!! note
    
    Switching to a Krylov linear solver will automatically change the nonlinear problem
    solver into Jacobian-free mode, dramatically reducing the memory required. This can be
    overridden by adding `concrete_jac=true` to the algorithm.

## Adding a Preconditioner

Any [LinearSolve.jl-compatible preconditioner](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/)
can be used as a preconditioner in the linear solver interface. To define preconditioners,
one must define a `precs` function in compatible with linear solvers which returns the
left and right preconditioners, matrices which approximate the inverse of `W = I - gamma*J`
used in the solution of the ODE. An example of this with using
[IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl) is as follows:

```julia
# FIXME: On 1.10+ this is broken. Skipping this for now.
using IncompleteLU

incompletelu(W, p = nothing) = ilu(W, τ = 50.0), LinearAlgebra.I

@btime solve(prob_brusselator_2d_sparse,
    NewtonRaphson(linsolve = KrylovJL_GMRES(precs = incompletelu), concrete_jac = true)
);
nothing # hide
```

Notice a few things about this preconditioner. This preconditioner uses the sparse Jacobian,
and thus we set `concrete_jac = true` to tell the algorithm to generate the Jacobian
(otherwise, a Jacobian-free algorithm is used with GMRES by default).

We use `convert(AbstractMatrix,W)` to get the concrete `W` matrix (matching `jac_prototype`,
thus `SpraseMatrixCSC`) which we can use in the preconditioner's definition. Then we use
`IncompleteLU.ilu` on that sparse matrix to generate the preconditioner. We return
`Pl, nothing` to say that our preconditioner is a left preconditioner, and that there is no
right preconditioning.

This method thus uses both the Krylov solver and the sparse Jacobian. Not only that, it is
faster than both implementations! IncompleteLU is fussy in that it requires a well-tuned `τ`
parameter. Another option is to use
[AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
which is more automatic. The setup is very similar to before:

```@example ill_conditioned_nlprob
using AlgebraicMultigrid

function algebraicmultigrid(W, p = nothing)
    return aspreconditioner(ruge_stuben(convert(AbstractMatrix, W))), LinearAlgebra.I
end

@btime solve(prob_brusselator_2d_sparse,
    NewtonRaphson(
        linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid), concrete_jac = true
    )
);
nothing # hide
```

or with a Jacobi smoother:

```@example ill_conditioned_nlprob
function algebraicmultigrid2(W, p = nothing)
    A = convert(AbstractMatrix, W)
    Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(
        A, presmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1))),
        postsmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1)))
    ))
    return Pl, LinearAlgebra.I
end

@btime solve(
    prob_brusselator_2d_sparse,
    NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = algebraicmultigrid2), concrete_jac = true
    )
);
nothing # hide
```

## Let's compare the Sparsity Detection Methods

We benchmarked the solvers before with approximate and exact sparsity detection. However,
for the exact sparsity detection case, we left out the time it takes to perform exact
sparsity detection. Let's compare the two by setting the sparsity detection algorithms.

```@example ill_conditioned_nlprob
using DifferentiationInterface, SparseConnectivityTracer

prob_brusselator_2d_exact_tracer = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop; sparsity = TracerSparsityDetector()),
    u0, p; abstol = 1e-10, reltol = 1e-10)
prob_brusselator_2d_approx_di = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop;
        sparsity = DenseSparsityDetector(AutoForwardDiff(); atol = 1e-4)),
    u0, p; abstol = 1e-10, reltol = 1e-10)

@btime solve(prob_brusselator_2d_exact_tracer, NewtonRaphson());
@btime solve(prob_brusselator_2d_approx_di, NewtonRaphson());
nothing # hide
```

For more information on the preconditioner interface, see the
[linear solver documentation](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/).
