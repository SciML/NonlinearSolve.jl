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
        no_implicit_imports = (;
            skip = (NonlinearSolveQuasiNewton, Base, Core, NonlinearSolveQuasiNewton.SciMLOperators),
        ),
        # @SciMLMessage / AbstractVerbosityPreset / None are owned by SciMLLogging and
        # re-exported through NonlinearSolveBase (where they are imported from).
        all_explicit_imports_via_owners = (;
            ignore = (Symbol("@SciMLMessage"), :AbstractVerbosityPreset, :None),
        ),
        # Non-public names qualified-accessed from their owning packages; NonlinearSolve
        # sublibraries are built on NonlinearSolveBase's internal API by design:
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
        #   SciMLBase: __init, NonlinearAliasSpecifier
        #   ArrayInterface: can_setindex, fast_scalar_indexing, restructure
        #   CommonSolve: init, solve, solve!
        #   LinearAlgebra: AdjOrTransAbsVec
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :AdjOrTransAbsVec, :NonlinearAliasSpecifier,
                :NonlinearSolveDefaultInit, :__init, :callback_into_cache!, :can_setindex,
                :check_and_update!, :condition_number, :evaluate_f, :evaluate_f!,
                :fast_scalar_indexing, :get_abstol, :get_fu, :get_full_jacobian,
                :get_linear_solver, :get_reltol, :get_u, :init, :init_nonlinearsolve_trace,
                :init_termination_cache, :initial_jacobian_scaling_alpha,
                :jacobian_initialized_preinverted, :make_identity!!, :maybe_pinv!!,
                :maybe_pinv!!_workspace, :maybe_unaliased, :maybe_unwrap_prob_for_enzyme,
                :reinit!, :reinit_common!, :reinit_self!, :reset!, :reset_timer!,
                :restructure, :run_initialization!, :safe_similar, :safe_vec, :solve,
                :solve!, :standardize_forwarddiff_tag, :step!, :store_inverse_jacobian,
                :stores_full_jacobian, :unwrap_val,
            ),
        ),
        # Non-public names explicitly imported from their owning packages:
        #   NonlinearSolveBase: @SciMLMessage, @static_timeit,
        #     AbstractApproximateJacobianStructure, AbstractApproximateJacobianUpdateRule,
        #     AbstractApproximateJacobianUpdateRuleCache, AbstractDescentDirection,
        #     AbstractJacobianCache, AbstractJacobianInitialization,
        #     AbstractNonlinearSolveAlgorithm, AbstractNonlinearSolveCache,
        #     AbstractResetCondition, AbstractResetConditionCache, AbstractVerbosityPreset,
        #     None, Utils, get_timer_output, update_trace!
        #   SciMLBase: AbstractNonlinearProblem, NLStats, NoSpecialize
        #   SciMLOperators: AbstractSciMLOperator
        #   StaticArraysCore: StaticArray
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@SciMLMessage"), Symbol("@static_timeit"),
                :AbstractApproximateJacobianStructure, :AbstractApproximateJacobianUpdateRule,
                :AbstractApproximateJacobianUpdateRuleCache, :AbstractDescentDirection,
                :AbstractJacobianCache, :AbstractJacobianInitialization,
                :AbstractNonlinearProblem, :AbstractNonlinearSolveAlgorithm,
                :AbstractNonlinearSolveCache, :AbstractResetCondition,
                :AbstractResetConditionCache, :AbstractSciMLOperator, :AbstractVerbosityPreset,
                :NLStats, :NoSpecialize, :None, :StaticArray, :Utils, :get_timer_output,
                :update_trace!,
            ),
        ),
    ),
)
