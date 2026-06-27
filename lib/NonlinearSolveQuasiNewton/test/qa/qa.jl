using SciMLTesting, NonlinearSolveQuasiNewton, Test

run_qa(
    NonlinearSolveQuasiNewton;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators]),
        deps_compat = (; ignore = [:SciMLJacobianOperators]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # `using SciMLOperators: AbstractSciMLOperator` also brings the SciMLOperators
        # module name into scope as an implicit import.
        no_implicit_imports = (; ignore = (:SciMLOperators,)),
        # @SciMLMessage / AbstractVerbosityPreset / None are owned by SciMLLogging and
        # re-exported through NonlinearSolveBase (where they are imported from).
        all_explicit_imports_via_owners = (;
            ignore = (Symbol("@SciMLMessage"), :AbstractVerbosityPreset, :None),
        ),
        # Still non-public in their owning packages after the make-public round
        # (NonlinearSolveBase's own internal API; the sublibrary builds on it by design):
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, condition_number, evaluate_f, evaluate_f!, get_abstol,
        #     get_fu, get_full_jacobian, get_linear_solver, get_reltol, get_u, init,
        #     init_nonlinearsolve_trace, init_termination_cache,
        #     initial_jacobian_scaling_alpha, jacobian_initialized_preinverted,
        #     make_identity!!, maybe_pinv!!, maybe_pinv!!_workspace, maybe_unaliased,
        #     maybe_unwrap_prob_for_enzyme, NonlinearSolveDefaultInit, reinit!,
        #     reinit_common!, reinit_self!, reset!, reset_timer!, restructure,
        #     run_initialization!, safe_similar, safe_vec, solve!, standardize_forwarddiff_tag,
        #     step!, store_inverse_jacobian, stores_full_jacobian, unwrap_val
        #   SciMLBase: __init;  LinearAlgebra: AdjOrTransAbsVec
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :AdjOrTransAbsVec,
                :NonlinearSolveDefaultInit, :__init, :callback_into_cache!,
                :check_and_update!, :condition_number, :evaluate_f, :evaluate_f!,
                :get_abstol, :get_fu, :get_full_jacobian, :get_linear_solver, :get_reltol,
                :get_u, :init, :init_nonlinearsolve_trace, :init_termination_cache,
                :initial_jacobian_scaling_alpha, :jacobian_initialized_preinverted,
                :make_identity!!, :maybe_pinv!!, :maybe_pinv!!_workspace, :maybe_unaliased,
                :maybe_unwrap_prob_for_enzyme, :reinit!, :reinit_common!, :reinit_self!,
                :reset!, :reset_timer!, :restructure, :run_initialization!, :safe_similar,
                :safe_vec, :solve!, :standardize_forwarddiff_tag, :step!,
                :store_inverse_jacobian, :stores_full_jacobian, :unwrap_val,
            ),
        ),
        # Still non-public in their owning packages after the make-public round:
        #   NonlinearSolveBase: @SciMLMessage, @static_timeit,
        #     AbstractApproximateJacobianStructure, AbstractApproximateJacobianUpdateRule,
        #     AbstractApproximateJacobianUpdateRuleCache, AbstractDescentDirection,
        #     AbstractJacobianCache, AbstractJacobianInitialization,
        #     AbstractNonlinearSolveAlgorithm, AbstractNonlinearSolveCache,
        #     AbstractResetCondition, AbstractResetConditionCache, AbstractVerbosityPreset,
        #     None, Utils, get_timer_output, update_trace!
        #   SciMLBase: NoSpecialize;  SciMLOperators: AbstractSciMLOperator
        #   StaticArraysCore: StaticArray
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@SciMLMessage"), Symbol("@static_timeit"),
                :AbstractApproximateJacobianStructure, :AbstractApproximateJacobianUpdateRule,
                :AbstractApproximateJacobianUpdateRuleCache, :AbstractDescentDirection,
                :AbstractJacobianCache, :AbstractJacobianInitialization,
                :AbstractNonlinearSolveAlgorithm, :AbstractNonlinearSolveCache,
                :AbstractResetCondition, :AbstractResetConditionCache, :AbstractSciMLOperator,
                :AbstractVerbosityPreset, :NoSpecialize, :None, :StaticArray, :Utils,
                :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
