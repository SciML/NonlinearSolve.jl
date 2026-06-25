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
        # Non-public names qualified-accessed from their owning packages:
        #   SciMLBase: AbstractNonlinearAlgorithm, build_linear_solution, build_solution,
        #     strip_solution
        #   CommonSolve: solve
        #   Base: Cartesian
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractNonlinearAlgorithm, :Cartesian, :build_linear_solution,
                :build_solution, :solve, :strip_solution,
            ),
        ),
    ),
)
