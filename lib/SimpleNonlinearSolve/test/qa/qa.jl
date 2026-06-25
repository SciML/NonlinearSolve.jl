using SciMLTesting, SimpleNonlinearSolve, Test
import ReverseDiff, Tracker, StaticArrays, Zygote

run_qa(
    SimpleNonlinearSolve;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators]),
        deps_compat = (; ignore = [:SciMLJacobianOperators]),
        piracies = (;
            treat_as_own = [
                NonlinearProblem, NonlinearLeastSquaresProblem, IntervalNonlinearProblem,
            ],
        ),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # ImmutableNonlinearProblem is owned by SciMLBase and re-exported through
        # NonlinearSolveBase (where the Tracker extension imports it from).
        all_explicit_imports_via_owners = (; ignore = (:ImmutableNonlinearProblem,)),
        # Non-public names qualified-accessed from their owning packages (Tracker ext):
        #   Tracker: @grad, data, track
        #   ArrayInterface: aos_to_soa
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@grad"), :aos_to_soa, :data, :simplenonlinearsolve_solve_up,
                :track,
            ),
        ),
        # Non-public names explicitly imported from their owning packages (Tracker ext):
        #   NonlinearSolveBase: ImmutableNonlinearProblem, _solve_adjoint
        #   Tracker: TrackedReal
        #   SciMLBase: TrackerOriginator
        all_explicit_imports_are_public = (;
            ignore = (
                :ImmutableNonlinearProblem, :TrackedReal, :TrackerOriginator,
                :_solve_adjoint,
            ),
        ),
    ),
)
