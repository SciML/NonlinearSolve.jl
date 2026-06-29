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
        # Still non-public in their owning packages, across the main module and the
        # Tracker / ReverseDiff / ChainRulesCore extensions. __init / __solve dropped:
        # now public in SciMLBase.
        #   Tracker: @grad, data, track;  ReverseDiff: value;  Base: setindex
        #   NonlinearSolveBase(.Utils): can_setindex, Utils, build_null_solution, evaluate_f,
        #     evaluate_f!!, init_termination_cache, maybe_unaliased, restructure, safe_similar,
        #     safe_vec, unwrap_val
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@grad"), :data, :track, :value, :can_setindex,
                :Utils, :build_null_solution,
                :evaluate_f, :evaluate_f!!, :init_termination_cache, :maybe_unaliased,
                :restructure, :safe_similar, :safe_vec, :unwrap_val, :setindex,
                :simplenonlinearsolve_solve_up,
            ),
        ),
        # Still non-public in their owning packages, main module + Tracker / ReverseDiff /
        # ChainRulesCore exts. @SciMLMessage / AbstractVerbosityPreset dropped: now imported
        # directly from their owner SciMLLogging (public there).
        #   NonlinearSolveBase: AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        #     AbstractSafeNonlinearTerminationMode, _solve_adjoint
        #   LineSearch: AbstractLineSearchAlgorithm;  ForwardDiff: Dual
        #   StaticArraysCore: StaticArray;  Tracker: TrackedReal
        #   ReverseDiff: TrackedArray, TrackedReal
        #   SciMLBase: ImmutableNonlinearProblem, TrackerOriginator, ChainRulesOriginator,
        #     ReverseDiffOriginator
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractLineSearchAlgorithm,
                :AbstractNonlinearSolveAlgorithm, :AbstractNonlinearTerminationMode,
                :AbstractSafeBestNonlinearTerminationMode,
                :AbstractSafeNonlinearTerminationMode, :Dual,
                :ImmutableNonlinearProblem, :StaticArray, :TrackedReal, :TrackedArray,
                :TrackerOriginator, :ChainRulesOriginator, :ReverseDiffOriginator,
                :_solve_adjoint, :simplenonlinearsolve_solve_up,
            ),
        ),
    ),
)
