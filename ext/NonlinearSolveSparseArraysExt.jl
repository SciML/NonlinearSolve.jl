module NonlinearSolveSparseArraysExt

using SparseArrays: SparseArrays
using NonlinearSolve: NonlinearSolve

# =============================================================================
# NonlinearSolve SparseArrays Integration Extension
# =============================================================================

"""
This extension is automatically loaded when SparseArrays is explicitly imported
by the user. It enables:

1. Sparse matrix support in NonlinearFunction with jac_prototype
2. Efficient sparse automatic differentiation
3. Sparse linear algebra in Newton-type methods
4. Integration with SparseMatrixColorings.jl for matrix coloring

Key Features Enabled:
- Sparse Jacobian prototypes in NonlinearFunction
- Automatic sparse AD when sparsity is detected
- Memory-efficient storage for large sparse systems
- Specialized algorithms for sparse matrices

Usage:
```julia
using NonlinearSolve
using SparseArrays  # This loads the extension

# Create sparse jacobian prototype
sparsity_pattern = sparse([1, 2], [1, 2], [true, true])
f = NonlinearFunction(my_function; jac_prototype=sparsity_pattern)
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, NewtonRaphson())
```

The extension works by ensuring all SparseArrays-specific methods in 
NonlinearSolveBase are available when sparse functionality is needed.
"""

# No additional functionality needed at the main NonlinearSolve level
# All sparse-specific implementations are in NonlinearSolveBaseSparseArraysExt

end