using SciMLTesting, NonlinearSolveSciPy, Test
using JET

const NONLINEARSOLVE_DOCS_SRC = joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src")

const SCIPY_EXTERNAL_REEXPORTS = union(
    public_api_names(NonlinearSolveSciPy.SciMLBase),
    (:SciMLBase,),
)

run_qa(
    NonlinearSolveSciPy;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = NONLINEARSOLVE_DOCS_SRC,
        ignore = SCIPY_EXTERNAL_REEXPORTS,
        rendered_ignore = SCIPY_EXTERNAL_REEXPORTS,
    ),
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
        # AbstractNonlinearSolveAlgorithm is non-public in NonlinearSolveBase.
        all_explicit_imports_are_public = (; ignore = (:AbstractNonlinearSolveAlgorithm,)),
    ),
)
