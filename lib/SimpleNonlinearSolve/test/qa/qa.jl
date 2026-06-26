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
        # @SciMLMessage / AbstractVerbosityPreset (owner SciMLLogging) and
        # ImmutableNonlinearProblem (owner SciMLBase) are re-exported through
        # NonlinearSolveBase, where they are imported from.
        all_explicit_imports_via_owners = (;
            ignore = (
                Symbol("@SciMLMessage"), :AbstractVerbosityPreset,
                :ImmutableNonlinearProblem,
            ),
        ),
        # Still non-public in their owning packages (not covered by the public-API round),
        # across the main module and the Tracker / ReverseDiff / ChainRulesCore extensions:
        #   Tracker: @grad, data, track;  ReverseDiff/NonlinearSolveBase.Utils: value
        #   SciMLBase: NoInit, NonlinearAliasSpecifier, __init, __solve;  Base: setindex
        #   NonlinearSolveBase(.Utils): can_setindex, Utils, build_null_solution, evaluate_f,
        #     evaluate_f!!, init_termination_cache, maybe_unaliased, restructure, safe_similar,
        #     safe_vec, unwrap_val
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@grad"), :data, :track, :value, :can_setindex, :NoInit,
                :NonlinearAliasSpecifier, :__init, :__solve, :Utils, :build_null_solution,
                :evaluate_f, :evaluate_f!!, :init_termination_cache, :maybe_unaliased,
                :restructure, :safe_similar, :safe_vec, :unwrap_val, :setindex,
                :simplenonlinearsolve_solve_up,
            ),
        ),
        # Still non-public in their owning packages, main module + Tracker / ReverseDiff /
        # ChainRulesCore exts:
        #   NonlinearSolveBase: @SciMLMessage, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        #     AbstractSafeNonlinearTerminationMode, AbstractVerbosityPreset,
        #     ImmutableNonlinearProblem, _solve_adjoint
        #   LineSearch: AbstractLineSearchAlgorithm;  ForwardDiff: Dual
        #   StaticArraysCore: StaticArray;  Tracker: TrackedReal
        #   ReverseDiff: TrackedArray, value
        #   SciMLBase: TrackerOriginator, ChainRulesOriginator, ReverseDiffOriginator
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@SciMLMessage"), :AbstractLineSearchAlgorithm,
                :AbstractNonlinearSolveAlgorithm, :AbstractNonlinearTerminationMode,
                :AbstractSafeBestNonlinearTerminationMode,
                :AbstractSafeNonlinearTerminationMode, :AbstractVerbosityPreset, :Dual,
                :ImmutableNonlinearProblem, :StaticArray, :TrackedReal, :TrackedArray,
                :TrackerOriginator, :ChainRulesOriginator, :ReverseDiffOriginator,
                :_solve_adjoint, :value, :simplenonlinearsolve_solve_up,
            ),
        ),
    ),
)
