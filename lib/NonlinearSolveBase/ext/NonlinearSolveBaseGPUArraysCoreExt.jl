module NonlinearSolveBaseGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray
using NonlinearSolveBase: NonlinearSolveBase

# For GPU arrays, when no specific linear solver is provided, we need to use normal form
# (J^T*J) for non-square Jacobians. This is because LinearSolve.jl's DefaultLinearSolver
# for GPU arrays tries to use Cholesky, which requires a square matrix.
# See https://github.com/SciML/NonlinearSolve.jl/issues/746
NonlinearSolveBase.needs_square_A(::Nothing, ::AbstractGPUArray) = true

end
