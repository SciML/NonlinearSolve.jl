using SciMLTesting, NonlinearSolveSpectralMethods, Test

run_qa(
    NonlinearSolveSpectralMethods;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators]),
        deps_compat = (; ignore = [:SciMLJacobianOperators]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # AbstractVerbosityPreset / None are owned by SciMLLogging and re-exported
        # through NonlinearSolveBase (where they are imported from).
        all_explicit_imports_via_owners = (;
            ignore = (:AbstractVerbosityPreset, :None),
        ),
        # Non-public names qualified-accessed from their owning packages; NonlinearSolve
        # sublibraries are built on NonlinearSolveBase's internal API by design:
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, evaluate_f, evaluate_f!, get_fu, get_u, init_nonlinearsolve_trace,
        #     init_termination_cache, maybe_unaliased, NonlinearSolveDefaultInit, reinit!,
        #     reinit_common!, reinit_self!, reset!, reset_timer!, run_initialization!, safe_dot,
        #     solve!, standardize_forwarddiff_tag, step!
        #   SciMLBase: __init, NonlinearAliasSpecifier
        #   CommonSolve: init, solve
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :NonlinearAliasSpecifier,
                :NonlinearSolveDefaultInit, :__init, :callback_into_cache!,
                :check_and_update!, :evaluate_f, :evaluate_f!, :get_fu, :get_u, :init,
                :init_nonlinearsolve_trace, :init_termination_cache, :maybe_unaliased,
                :reinit!, :reinit_common!, :reinit_self!, :reset!, :reset_timer!,
                :run_initialization!, :safe_dot, :solve, :solve!,
                :standardize_forwarddiff_tag, :step!,
            ),
        ),
        # Non-public names explicitly imported from their owning packages:
        #   NonlinearSolveBase: @static_timeit, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearSolveCache, AbstractVerbosityPreset, None, Utils,
        #     get_timer_output, update_trace!
        #   SciMLBase: AbstractNonlinearProblem, NLStats, NoSpecialize
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@static_timeit"), :AbstractNonlinearProblem,
                :AbstractNonlinearSolveAlgorithm, :AbstractNonlinearSolveCache,
                :AbstractVerbosityPreset, :NLStats, :NoSpecialize, :None, :Utils,
                :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
