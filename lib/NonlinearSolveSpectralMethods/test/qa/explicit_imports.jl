using NonlinearSolveSpectralMethods

using ExplicitImports, NonlinearSolveSpectralMethods

@test check_no_implicit_imports(
    NonlinearSolveSpectralMethods; skip = (Base, Core, SciMLBase)
) === nothing
@test check_no_stale_explicit_imports(NonlinearSolveSpectralMethods) === nothing
@test check_all_qualified_accesses_via_owners(NonlinearSolveSpectralMethods) === nothing
