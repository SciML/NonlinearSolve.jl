using SciMLTesting, NonlinearSolveHomotopyContinuation, Test

run_qa(
    NonlinearSolveHomotopyContinuation;
    explicit_imports = true,
    # persistent_tasks: intermittently errors on Julia >=1.11 because the registered
    # NonlinearSolveBase ships a leaked `[sources]` (SciMLJacobianOperators =
    # {path = "../SciMLJacobianOperators"}) that Pkg >=1.11 honors during Aqua's
    # develop-by-path check, so the sibling path does not resolve. This is the upstream
    # Pkg [sources]-leak bug (JuliaLang/Pkg.jl#4705 + JuliaTesting/Aqua.jl#387), not a
    # real persistent task; it flakes the same way on master. Marked broken so the lane
    # is stable; drop when the Pkg/Aqua fix lands. (This env does not develop
    # SciMLJacobianOperators, so run_qa's clean_sources cannot reach the leaked entry.)
    aqua_broken = (:persistent_tasks,),
    ei_kwargs = (;
        # Still non-public in their owning packages after the make-public round:
        #   NonlinearSolveBase(.Utils): AbstractNonlinearSolveAlgorithm, Utils, evaluate_f
        #   TaylorDiff: flatten, value
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractNonlinearSolveAlgorithm, :Utils, :evaluate_f, :flatten, :value,
            ),
        ),
    ),
    # no_implicit_imports: ~20 names reach the module via `using` of ADTypes,
    # DocStringExtensions, LinearAlgebra, NonlinearSolveBase, SciMLBase,
    # SymbolicIndexingInterface and TaylorDiff. Making each explicit is a large refactor
    # tracked separately (this QA PR); kept broken so the lane records Broken, not Fail.
    ei_broken = (:no_implicit_imports,),
)
