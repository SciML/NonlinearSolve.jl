using NonlinearSolveQuasiNewton

using ExplicitImports, NonlinearSolveQuasiNewton

@test check_no_implicit_imports(
    NonlinearSolveQuasiNewton; skip = (Base, Core, SciMLBase)
) === nothing
@test check_no_stale_explicit_imports(NonlinearSolveQuasiNewton) === nothing
@test check_all_qualified_accesses_via_owners(NonlinearSolveQuasiNewton) === nothing
