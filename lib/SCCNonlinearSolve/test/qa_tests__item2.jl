using SCCNonlinearSolve

using ExplicitImports, SciMLBase, SCCNonlinearSolve

@test check_no_implicit_imports(
    SCCNonlinearSolve; skip = (Base, Core, SciMLBase)
) === nothing
@test check_no_stale_explicit_imports(SCCNonlinearSolve) === nothing
@test check_all_qualified_accesses_via_owners(SCCNonlinearSolve) === nothing
