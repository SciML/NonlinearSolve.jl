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
        # Still non-public in their owning packages (NonlinearSolveBase's own internal API;
        # the sublibrary builds on it by design). __init dropped: now public in SciMLBase.
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, evaluate_f, evaluate_f!, get_fu, get_u,
        #     init_nonlinearsolve_trace, init_termination_cache, maybe_unaliased,
        #     NonlinearSolveDefaultInit, reinit!, reinit_common!, reinit_self!, reset!,
        #     reset_timer!, run_initialization!, safe_dot, standardize_forwarddiff_tag, step!
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :NonlinearSolveDefaultInit,
                :callback_into_cache!, :check_and_update!, :evaluate_f, :evaluate_f!,
                :get_fu, :get_u, :init_nonlinearsolve_trace, :init_termination_cache,
                :maybe_unaliased, :reinit!, :reinit_common!, :reinit_self!, :reset!,
                :reset_timer!, :run_initialization!, :safe_dot, :standardize_forwarddiff_tag,
                :step!,
            ),
        ),
        # Still non-public in their owning packages. AbstractVerbosityPreset / None dropped:
        # now imported directly from their owner SciMLLogging (public there).
        #   NonlinearSolveBase: @static_timeit, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearSolveCache, Utils, get_timer_output, update_trace!
        #   SciMLBase: NoSpecialize
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@static_timeit"), :AbstractNonlinearSolveAlgorithm,
                :AbstractNonlinearSolveCache, :NoSpecialize,
                :Utils, :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
