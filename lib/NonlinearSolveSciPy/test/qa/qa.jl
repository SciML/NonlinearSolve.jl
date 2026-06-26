using SciMLTesting, NonlinearSolveSciPy, Test
using JET

run_qa(
    NonlinearSolveSciPy;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
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
        # CommonSolve / NonlinearSolveBase / SciMLBase module names reach the module
        # via `using X: name` forms (the module name itself is an implicit import).
        no_implicit_imports = (;
            ignore = (:CommonSolve, :NonlinearSolveBase, :SciMLBase),
        ),
        # Still non-public in their owning packages (not covered by the public-API round):
        #   SciMLBase: __solve, allowsbounds
        all_qualified_accesses_are_public = (;
            ignore = (:__solve, :allowsbounds),
        ),
        # AbstractNonlinearSolveAlgorithm is non-public in NonlinearSolveBase.
        all_explicit_imports_are_public = (; ignore = (:AbstractNonlinearSolveAlgorithm,)),
    ),
)
