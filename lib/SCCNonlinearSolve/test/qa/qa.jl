using SciMLTesting, SCCNonlinearSolve, Test

const NONLINEARSOLVE_DOCS_SRC = joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src")

run_qa(
    SCCNonlinearSolve;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]),
        deps_compat = (; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]),
        piracies = (; treat_as_own = [SCCNonlinearSolve.SciMLBase.solve]),
        ambiguities = (; recursive = false),
    ),
    api_docs_kwargs = (; rendered = true, docs_src = NONLINEARSOLVE_DOCS_SRC),
    ei_kwargs = (;
        # Still non-public in their owning packages after the make-public round:
        #   SciMLBase: build_linear_solution, strip_solution;  Base: Cartesian
        all_qualified_accesses_are_public = (;
            ignore = (:Cartesian, :build_linear_solution, :strip_solution),
        ),
    ),
)
