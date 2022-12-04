# [Nonlinear System Solvers](@id nonlinearsystemsolvers)

`solve(prob::NonlinearProblem,alg;kwargs)`

Solves for ``f(u)=0`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

This page is solely focused on the methods for nonlinear systems.

## Recommended Methods

`NewtonRaphson` is a good choice for most problems.  For large
systems it can make use of sparsity patterns for sparse automatic differentiation
and sparse linear solving of very large systems. That said, as a classic Newton
method, its stability region can be smaller than other methods. Meanwhile, `SimpleNewtonRaphson`
is an implementation which is specialized for small equations. It is non-allocating on
static arrays and thus really well-optimized for small systems, thus usually outperforming
the other methods when such types are used for `u0`. `NLSolveJL`'s
`:trust_region` method can be a good choice for high stability, along with
`DynamicSS`.

For a system which is very non-stiff (i.e., the condition number of the Jacobian
is small, or the eigenvalues of the Jacobian are within a few orders of magnitude),
then `NLSolveJL`'s `:anderson` can be a good choice.

## Full List of Methods

### NonlinearSolve.jl

These are the core solvers.

- `NewtonRaphson()`:
  A Newton-Raphson method with swappable nonlinear solvers and autodiff methods
  for high performance on large and sparse systems.

#### Details on Controlling NonlinearSolve.jl Solvers

```julia
NewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
                standardtag = Val{true}(), concrete_jac = nothing,
                diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS)
```

### SimpleNonlinearSolve.jl

These methods are included with NonlinearSolve.jl by default, though SimpleNonlinearSolve.jl
can be used directly to reduce dependencies and improve load times.

- `SimpleNewtonRaphson()`: A simplified implementation of the Newton-Raphson method. Has the
  property that when used with states `u` as a `Number` or `StaticArray`, the solver is
  very efficient and non-allocating. Thus this implmentation is well-suited for small
  systems of equations.

### SteadyStateDiffEq.jl

Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
]add SteadyStateDiffEq
using SteadyStateDiffEq
```

SteadyStateDiffEq.jl uses ODE solvers to iteratively approach the steady state. It is a
very stable method for solving nonlinear systems, though in many cases can be more
computationally expensive than direct methods.

- `DynamicSS` : Uses an ODE solver to find the steady state. Automatically
  terminates when close to the steady state.
  `DynamicSS(alg;abstol=1e-8,reltol=1e-6,tspan=Inf)` requires that an
  ODE algorithm is given as the first argument.  The absolute and
  relative tolerances specify the termination conditions on the
  derivative's closeness to zero.  This internally uses the
  `TerminateSteadyState` callback from the Callback Library.  The
  simulated time for which given ODE is solved can be limited by
  `tspan`.  If `tspan` is a number, it is equivalent to passing
  `(zero(tspan), tspan)`.

Example usage:

```julia
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
sol = solve(prob,DynamicSS(Tsit5()))

using Sundials
sol = solve(prob,DynamicSS(CVODE_BDF()),dt=1.0)
```

!!! note

    If you use `CVODE_BDF` you may need to give a starting `dt` via `dt=....`.*

### SciMLNLSolve.jl

This is a wrapper package for importing solvers from NLsolve.jl into this interface.
Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
]add SciMLNLSolve
using SciMLNLSolve
```

- `NLSolveJL()`: A wrapper for [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)

```julia
NLSolveJL(;
          method=:trust_region,
          autodiff=:central,
          store_trace=false,
          extended_trace=false,
          linesearch=LineSearches.Static(),
          linsolve=(x, A, b) -> copyto!(x, A\b),
          factor = one(Float64),
          autoscale=true,
          m=10,
          beta=one(Float64),
          show_trace=false,
       )
```

Choices for methods in `NLSolveJL`:

- `:fixedpoint`: Fixed-point iteration
- `:anderson`: Anderson-accelerated fixed-point iteration
- `:newton`: Classical Newton method with an optional line search
- `:trust_region`: Trust region Newton method (the default choice)

For more information on these arguments, consult the
[NLsolve.jl documentation](https://github.com/JuliaNLSolvers/NLsolve.jl).

### Sundials.jl

This is a wrapper package for the SUNDIALS C library, specifically the KINSOL
nonlinear solver included in that ecosystem. Note that these solvers do not come
by default, and thus one needs to install the package before using these solvers:

```julia
]add Sundials
using Sundials
```

- `KINSOL`: The KINSOL method of the SUNDIALS C library

```julia
KINSOL(;
    linear_solver = :Dense,
    jac_upper = 0,
    jac_lower = 0,
    userdata = nothing,
)
```

The choices for the linear solver are:

- `:Dense`: A dense linear solver
- `:Band`: A solver specialized for banded Jacobians. If used, you must set the
  position of the upper and lower non-zero diagonals via `jac_upper` and
  `jac_lower`.
- `:LapackDense`: A version of the dense linear solver that uses the Julia-provided
  OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than
  `:Dense` on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
- `:LapackBand`: A version of the banded linear solver that uses the Julia-provided
  OpenBLAS-linked LAPACK for multithreaded operations. This will be faster than
  `:Band` on larger systems but has noticeable overhead on smaller (<100 ODE) systems.
- `:Diagonal`: This method is specialized for diagonal Jacobians.
- `:GMRES`: A GMRES method. Recommended first choice Krylov method.
- `:BCG`: A biconjugate gradient method
- `:PCG`: A preconditioned conjugate gradient method. Only for symmetric
  linear systems.
- `:TFQMR`: A TFQMR method.
- `:KLU`: A sparse factorization method. Requires that the user specify a
  Jacobian. The Jacobian must be set as a sparse matrix in the `ODEProblem`
  type.

### NonlinearSolveMINPACK

This is a wrapper package for importing solvers from MINPACK.jl into this interface.
Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
]add NonlinearSolveMINPACK
using NonlinearSolveMINPACK
```

- `CMINPACK()`: A wrapper for using the classic MINPACK method through [MINPACK.jl](https://github.com/sglyon/MINPACK.jl)

```julia
CMINPACK(;show_trace::Bool=false, tracing::Bool=false, method::Symbol=:hybr,
          io::IO=stdout)
```

The keyword argument `method` can take on different value depending on which method of `fsolve` you are calling. The standard choices of `method` are:

- `:hybr`: Modified version of Powell's algorithm. Uses MINPACK routine [`hybrd1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd1.c)
- `:lm`: Levenberg-Marquardt. Uses MINPACK routine [`lmdif1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif1.c)
- `:lmdif`: Advanced Levenberg-Marquardt (more options available with `;kwargs...`). See MINPACK routine [`lmdif`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif.c) for more information
- `:hybrd`: Advacned modified version of Powell's algorithm (more options available with `;kwargs...`). See MINPACK routine [`hybrd`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd.c) for more information

If a Jacobian is supplied as part of the [`NonlinearFunction`](@ref nonlinearfunctions),
then the following methods are allowed:

- `:hybr`: Advacned modified version of Powell's algorithm with user supplied Jacobian. Additional arguments are available via `;kwargs...`. See MINPACK routine [`hybrj`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrj.c) for more information
- `:lm`: Advanced Levenberg-Marquardt with user supplied Jacobian. Additional arguments are available via `;kwargs...`. See MINPACK routine [`lmder`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmder.c) for more information
