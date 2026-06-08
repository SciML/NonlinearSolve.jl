import ForwardDiff, SparseArrays
using ExplicitImports, NonlinearSolveBase

@test check_no_implicit_imports(NonlinearSolveBase; skip = (Base, Core)) === nothing
# Ignore SciMLLogging types used by @verbosity_specifier macro (ExplicitImports can't track macro usage)
@test check_no_stale_explicit_imports(
    NonlinearSolveBase;
    ignore = (:MessageLevel, :AbstractVerbositySpecifier, :All, :Detailed, :Minimal, :None, :Standard, :SciMLLogging)
) === nothing
@test check_all_qualified_accesses_via_owners(NonlinearSolveBase) === nothing
