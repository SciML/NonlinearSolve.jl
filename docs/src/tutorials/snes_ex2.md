# [PETSc SNES Example 2](@id snes_ex2)

This implements `src/snes/examples/tutorials/ex2.c` from PETSc and `examples/SNES_ex2.jl`
from PETSc.jl using automatic sparsity detection and automatic differentiation using
`NonlinearSolve.jl`.

This solves the equations sequentially. Newton method to solve
`u'' + u^{2} = f`, sequentially.

```@example snes_ex2
import NonlinearSolve as NLS
import PETSc
import LinearAlgebra
import SparseConnectivityTracer
import BenchmarkTools

u0 = fill(0.5, 128)

function form_residual!(resid, x, _)
    n = length(x)
    xp = LinRange(0.0, 1.0, n)
    F = 6xp .+ (xp .+ 1e-12) .^ 6

    dx = 1 / (n - 1)
    resid[1] = x[1]
    for i in 2:(n - 1)
        resid[i] = (x[i - 1] - 2x[i] + x[i + 1]) / dx^2 + x[i] * x[i] - F[i]
    end
    resid[n] = x[n] - 1

    return
end
```

To use automatic sparsity detection, we need to specify `sparsity` keyword argument to
`NonlinearFunction`. See [Automatic Sparsity Detection](@ref sparsity-detection) for more
details.

```@example snes_ex2
nlfunc_dense = NLS.NonlinearFunction(form_residual!)
nlfunc_sparse = NLS.NonlinearFunction(
    form_residual!; sparsity = SparseConnectivityTracer.TracerSparsityDetector())

nlprob_dense = NLS.NonlinearProblem(nlfunc_dense, u0)
nlprob_sparse = NLS.NonlinearProblem(nlfunc_sparse, u0)
```

Now we can solve the problem using `PETScSNES` or with one of the native `NonlinearSolve.jl`
solvers.

```@example snes_ex2
sol_dense_nr = NLS.solve(nlprob_dense, NLS.NewtonRaphson(); abstol = 1e-8)
sol_dense_snes = NLS.solve(nlprob_dense, NLS.PETScSNES(); abstol = 1e-8)
sol_dense_nr .- sol_dense_snes
```

```@example snes_ex2
sol_sparse_nr = NLS.solve(nlprob_sparse, NLS.NewtonRaphson(); abstol = 1e-8)
sol_sparse_snes = NLS.solve(nlprob_sparse, NLS.PETScSNES(); abstol = 1e-8)
sol_sparse_nr .- sol_sparse_snes
```

As expected the solutions are the same (upto floating point error). Now let's compare the
runtimes.

## Runtimes

### Dense Jacobian

```@example snes_ex2
BenchmarkTools.@benchmark NLS.solve($(nlprob_dense), $(NLS.NewtonRaphson()); abstol = 1e-8)
```

```@example snes_ex2
BenchmarkTools.@benchmark NLS.solve($(nlprob_dense), $(NLS.PETScSNES()); abstol = 1e-8)
```

### Sparse Jacobian

```@example snes_ex2
BenchmarkTools.@benchmark NLS.solve($(nlprob_sparse), $(NLS.NewtonRaphson()); abstol = 1e-8)
```

```@example snes_ex2
BenchmarkTools.@benchmark NLS.solve($(nlprob_sparse), $(NLS.PETScSNES()); abstol = 1e-8)
```
