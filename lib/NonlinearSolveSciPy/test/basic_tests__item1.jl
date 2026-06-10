using NonlinearSolveSciPy

using Test, NonlinearSolveSciPy
@test isdefined(NonlinearSolveSciPy, :SciPyLeastSquares)
@test isdefined(NonlinearSolveSciPy, :SciPyRoot)
