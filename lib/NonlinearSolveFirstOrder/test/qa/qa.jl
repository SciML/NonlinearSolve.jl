using SciMLTesting, NonlinearSolveFirstOrder, Test

run_qa(
    NonlinearSolveFirstOrder;
    explicit_imports = true,
    aqua_kwargs = (;
        piracies = (; treat_as_own = [NonlinearLeastSquaresProblem]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # CommonSolve.init is used unqualified in forward_diff.jl (the
        # CommonSolve.solve method dispatch surfaces `init` as an implicit import).
        no_implicit_imports = (; ignore = (:init,)),
        # @SciMLMessage / AbstractVerbosityPreset / None are owned by SciMLLogging and
        # re-exported through NonlinearSolveBase (where they are imported from).
        all_explicit_imports_via_owners = (;
            ignore = (Symbol("@SciMLMessage"), :AbstractVerbosityPreset, :None),
        ),
        # Still non-public in their owning packages (NonlinearSolveBase's own internal API
        # was not covered by the public-API round); the sublibrary builds on it by design:
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, evaluate_f, evaluate_f!, evaluate_f!!, get_fu,
        #     get_linear_solver, get_u, init, init_nonlinearsolve_trace, init_termination_cache,
        #     initialization_alg, last_step_accepted, maybe_unaliased,
        #     maybe_unwrap_prob_for_enzyme, NewtonDescentCache, NonlinearSolveDefaultInit,
        #     nodual_value, reinit!, reinit_common!, reinit_self!,
        #     requires_normal_form_jacobian, requires_normal_form_rhs, reset!, reset_timer!,
        #     returns_norm_form_damping, run_initialization!, safe_dot, safe_vec, solve!,
        #     standardize_forwarddiff_tag, step!
        #   SciMLBase: __init, __solve, NonlinearAliasSpecifier
        #   ForwardDiff: partials;  LinearSolve: update_tolerances!
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :NewtonDescentCache, :NonlinearAliasSpecifier,
                :NonlinearSolveDefaultInit, :__init, :__solve, :callback_into_cache!,
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
        # Still non-public in their owning packages:
        #   NonlinearSolveBase: @SciMLMessage, @static_timeit, AbstractDampingFunction,
        #     AbstractDampingFunctionCache, AbstractNonlinearSolveAlgorithm,
        #     AbstractNonlinearSolveCache, AbstractTrustRegionMethod,
        #     AbstractTrustRegionMethodCache, AbstractVerbosityPreset, None,
        #     NonlinearSolveForwardDiffCache, Utils, get_timer_output, update_trace!
        #   SciMLBase: NoSpecialize;  ForwardDiff: Dual
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@SciMLMessage"), Symbol("@static_timeit"), :AbstractDampingFunction,
                :AbstractDampingFunctionCache, :AbstractNonlinearSolveAlgorithm,
                :AbstractNonlinearSolveCache, :AbstractTrustRegionMethod,
                :AbstractTrustRegionMethodCache, :AbstractVerbosityPreset, :Dual,
                :NoSpecialize, :None, :NonlinearSolveForwardDiffCache, :Utils,
                :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
