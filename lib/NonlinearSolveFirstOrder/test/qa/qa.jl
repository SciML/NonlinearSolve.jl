using SciMLTesting, NonlinearSolveFirstOrder, Test

run_qa(
    NonlinearSolveFirstOrder;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (; treat_as_own = [NonlinearLeastSquaresProblem]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # Still non-public in their owning packages (NonlinearSolveBase's own internal API;
        # the sublibrary builds on it by design). __init / __solve dropped: now public in
        # SciMLBase.
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, evaluate_f, evaluate_f!, evaluate_f!!, get_fu,
        #     get_linear_solver, get_u, init, init_nonlinearsolve_trace, init_termination_cache,
        #     initialization_alg, last_step_accepted, maybe_unaliased,
        #     maybe_unwrap_prob_for_enzyme, NewtonDescentCache, NonlinearSolveDefaultInit,
        #     nodual_value, reinit!, reinit_common!, reinit_self!,
        #     requires_normal_form_jacobian, requires_normal_form_rhs, reset!, reset_timer!,
        #     returns_norm_form_damping, run_initialization!, safe_dot, safe_vec, solve!,
        #     standardize_forwarddiff_tag, step!
        #   ForwardDiff: partials;  LinearSolve: update_tolerances!
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :NewtonDescentCache,
                :NonlinearSolveDefaultInit, :callback_into_cache!,
                :check_and_update!, :evaluate_f, :evaluate_f!, :evaluate_f!!, :get_fu,
                :get_linear_solver, :get_u, :init, :init_nonlinearsolve_trace,
                :init_termination_cache, :initialization_alg, :last_step_accepted,
                :maybe_unaliased, :maybe_unwrap_prob_for_enzyme, :nodual_value, :partials,
                :reinit!, :reinit_common!, :reinit_self!, :requires_normal_form_jacobian,
                :requires_normal_form_rhs, :reset!, :reset_timer!, :returns_norm_form_damping,
                :run_initialization!, :safe_dot, :safe_vec, :solve!,
                :standardize_forwarddiff_tag, :step!, :update_tolerances!,
            ),
        ),
        # Still non-public in their owning packages. @SciMLMessage / AbstractVerbosityPreset
        # / None dropped: now imported directly from their owner SciMLLogging (public there).
        #   NonlinearSolveBase: @static_timeit, AbstractDampingFunction,
        #     AbstractDampingFunctionCache, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearSolveCache, AbstractTrustRegionMethod,
        #     AbstractTrustRegionMethodCache, NonlinearSolveForwardDiffCache, Utils,
        #     get_timer_output, update_trace!
        #   SciMLBase: NoSpecialize;  ForwardDiff: Dual
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@static_timeit"), :AbstractDampingFunction,
                :AbstractDampingFunctionCache, :AbstractNonlinearSolveAlgorithm,
                :AbstractNonlinearSolveCache, :AbstractTrustRegionMethod,
                :AbstractTrustRegionMethodCache, :Dual,
                :NoSpecialize, :NonlinearSolveForwardDiffCache, :Utils,
                :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
