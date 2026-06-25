using SciMLTesting, NonlinearSolveSciPy, Test
using JET

run_qa(
    NonlinearSolveSciPy;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # CommonSolve / NonlinearSolveBase / SciMLBase module names reach the module
        # via `using X: name` forms (the module name itself is an implicit import).
        no_implicit_imports = (;
            skip = (
                NonlinearSolveSciPy, Base, Core,
                NonlinearSolveSciPy.CommonSolve, NonlinearSolveSciPy.NonlinearSolveBase,
                NonlinearSolveSciPy.SciMLBase,
            ),
        ),
        # Non-public names qualified-accessed from their owning packages:
        #   SciMLBase: NLStats, __solve, allowsbounds, build_solution
        #   CommonSolve: solve
        all_qualified_accesses_are_public = (;
            ignore = (:NLStats, :__solve, :allowsbounds, :build_solution, :solve),
        ),
        # AbstractNonlinearSolveAlgorithm is non-public in NonlinearSolveBase.
        all_explicit_imports_are_public = (; ignore = (:AbstractNonlinearSolveAlgorithm,)),
    ),
)
