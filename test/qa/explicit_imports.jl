using NonlinearSolve

using NonlinearSolve, ADTypes, SimpleNonlinearSolve, SciMLBase
import FastLevenbergMarquardt, FixedPointAcceleration, LeastSquaresOptim, MINPACK,
    NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping

using ExplicitImports

@test check_no_implicit_imports(NonlinearSolve; skip = (Base, Core)) === nothing
@test check_no_stale_explicit_imports(NonlinearSolve) === nothing
@test check_all_qualified_accesses_via_owners(NonlinearSolve) === nothing
