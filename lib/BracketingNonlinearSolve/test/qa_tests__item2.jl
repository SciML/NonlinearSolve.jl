using BracketingNonlinearSolve

import ForwardDiff
using ExplicitImports, BracketingNonlinearSolve

@test check_no_implicit_imports(BracketingNonlinearSolve; skip = (Base, Core)) ===
    nothing
@test check_no_stale_explicit_imports(BracketingNonlinearSolve) === nothing
@test check_all_qualified_accesses_via_owners(BracketingNonlinearSolve) === nothing
