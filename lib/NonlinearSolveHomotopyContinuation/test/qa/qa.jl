using SciMLTesting, NonlinearSolveHomotopyContinuation, Test

run_qa(
    NonlinearSolveHomotopyContinuation;
    explicit_imports = true,
    ei_kwargs = (;
        # Non-public names qualified-accessed from their owning packages:
        #   NonlinearSolveBase(.Utils): AbstractNonlinearSolveAlgorithm, Utils, evaluate_f
        #   SciMLBase: build_solution, has_jac
        #   TaylorDiff: flatten, value
        #   CommonSolve: solve
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractNonlinearSolveAlgorithm, :Utils, :build_solution, :evaluate_f,
                :flatten, :has_jac, :solve, :value,
            ),
        ),
    ),
    # no_implicit_imports: ~20 names reach the module via `using` of ADTypes,
    # DocStringExtensions, LinearAlgebra, NonlinearSolveBase, SciMLBase,
    # SymbolicIndexingInterface and TaylorDiff. Making each explicit is a large refactor
    # tracked separately (this QA PR); kept broken so the lane records Broken, not Fail.
    ei_broken = (:no_implicit_imports,),
)
