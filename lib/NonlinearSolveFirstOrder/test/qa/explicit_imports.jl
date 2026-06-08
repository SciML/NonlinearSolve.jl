using NonlinearSolveFirstOrder

using ExplicitImports, NonlinearSolveFirstOrder

@test check_no_implicit_imports(
    NonlinearSolveFirstOrder; skip = (Base, Core, SciMLBase)
) === nothing
@test check_no_stale_explicit_imports(NonlinearSolveFirstOrder) === nothing
@test check_all_qualified_accesses_via_owners(NonlinearSolveFirstOrder) === nothing
