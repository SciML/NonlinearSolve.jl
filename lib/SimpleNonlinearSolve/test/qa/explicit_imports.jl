using SimpleNonlinearSolve

import ReverseDiff, Tracker, StaticArrays, Zygote
using ExplicitImports, SimpleNonlinearSolve

@test check_no_implicit_imports(SimpleNonlinearSolve; skip = (Base, Core)) === nothing
@test check_no_stale_explicit_imports(SimpleNonlinearSolve) === nothing
@test check_all_qualified_accesses_via_owners(SimpleNonlinearSolve) === nothing
