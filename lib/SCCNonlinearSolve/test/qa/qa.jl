using SciMLTesting, SCCNonlinearSolve, Test

run_qa(
    SCCNonlinearSolve;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]),
        deps_compat = (; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]),
        piracies = (; treat_as_own = [SCCNonlinearSolve.SciMLBase.solve]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # Still non-public in their owning packages (not covered by the public-API round):
        #   SciMLBase: AbstractNonlinearAlgorithm, build_linear_solution, strip_solution
        #   Base: Cartesian
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractNonlinearAlgorithm, :Cartesian, :build_linear_solution,
                :strip_solution,
            ),
        ),
    ),
)
