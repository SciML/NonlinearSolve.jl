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
        # Non-public names qualified-accessed from their owning packages, across the
        # main module and the Tracker / ReverseDiff / ChainRulesCore extensions:
        #   Tracker: @grad, data, track;  ArrayInterface: aos_to_soa, can_setindex,
        #     fast_scalar_indexing
        #   SciMLBase: AbstractNonlinearProblem, NoInit, NonlinearAliasSpecifier, __init,
        #     __solve, build_solution, has_jac, has_jvp, has_vjp
        #   NonlinearSolveBase(.Utils): Utils, build_null_solution, evaluate_f,
        #     evaluate_f!!, init_termination_cache, maybe_unaliased, restructure,
        #     safe_similar, safe_vec, unwrap_val
        #   CommonSolve: solve;  Base: setindex;  ReverseDiff: value
        #   SimpleNonlinearSolve (own internal): simplenonlinearsolve_solve_up
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@grad"), :aos_to_soa, :can_setindex, :data, :track, :value,
                :fast_scalar_indexing, :AbstractNonlinearProblem, :NoInit,
                :NonlinearAliasSpecifier, :__init, :__solve, :build_solution, :has_jac,
                :has_jvp, :has_vjp, :Utils, :build_null_solution, :evaluate_f,
                :evaluate_f!!, :init_termination_cache, :maybe_unaliased, :restructure,
                :safe_similar, :safe_vec, :unwrap_val, :solve, :setindex,
                :simplenonlinearsolve_solve_up,
            ),
        ),
        # Non-public names explicitly imported from their owning packages, main module
        # + Tracker / ReverseDiff / ChainRulesCore exts:
        #   NonlinearSolveBase: @SciMLMessage, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        #     AbstractSafeNonlinearTerminationMode, AbstractVerbosityPreset,
        #     ImmutableNonlinearProblem, _solve_adjoint
        #   LineSearch: AbstractLineSearchAlgorithm;  ForwardDiff: Dual
        #   StaticArraysCore: StaticArray;  CommonSolve: init, solve, solve!
        #   Tracker: TrackedReal;  ReverseDiff: value, TrackedArray
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
                :_solve_adjoint, :init, :solve, :solve!, :value,
                :simplenonlinearsolve_solve_up,
            ),
        ),
    ),
)
