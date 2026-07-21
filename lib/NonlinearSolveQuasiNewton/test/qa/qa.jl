using SciMLTesting, NonlinearSolveQuasiNewton, Test

const NONLINEARSOLVE_DOCS_SRC = joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src")

const QUASI_NEWTON_EXTERNAL_REEXPORTS = union(
    public_api_names(NonlinearSolveQuasiNewton.SciMLBase),
    (:SciMLBase,),
)

run_qa(
    NonlinearSolveQuasiNewton;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators]),
        deps_compat = (; ignore = [:SciMLJacobianOperators]),
        ambiguities = (; recursive = false),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = NONLINEARSOLVE_DOCS_SRC,
        ignore = QUASI_NEWTON_EXTERNAL_REEXPORTS,
        rendered_ignore = QUASI_NEWTON_EXTERNAL_REEXPORTS,
    ),
    ei_kwargs = (;
        # Still non-public in their owning packages (NonlinearSolveBase's own internal API;
        # the sublibrary builds on it by design). __init dropped: now public in SciMLBase.
        #   NonlinearSolveBase(.Utils/.InternalAPI): @internal_caches, callback_into_cache!,
        #     check_and_update!, condition_number, evaluate_f, evaluate_f!, get_abstol,
        #     get_fu, get_full_jacobian, get_linear_solver, get_reltol, get_u, init,
        #     init_nonlinearsolve_trace, init_termination_cache,
        #     initial_jacobian_scaling_alpha, linsolve_identity!!, linsolve_workspace,
        #     jacobian_initialized_preinverted, make_identity!!, maybe_unaliased,
        #     maybe_unwrap_prob_for_enzyme, NonlinearSolveDefaultInit, reinit!,
        #     reinit_common!, reinit_self!, reset!, reset_timer!, restructure,
        #     run_initialization!, safe_similar, safe_vec, solve!, standardize_forwarddiff_tag,
        #     step!, store_inverse_jacobian, stores_full_jacobian, unwrap_val
        #   LinearAlgebra: AdjOrTransAbsVec
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@internal_caches"), :AdjOrTransAbsVec,
                :NonlinearSolveDefaultInit, :callback_into_cache!,
                :check_and_update!, :condition_number, :evaluate_f, :evaluate_f!,
                :get_abstol, :get_fu, :get_full_jacobian, :get_linear_solver, :get_reltol,
                :get_u, :init, :init_nonlinearsolve_trace, :init_termination_cache,
                :initial_jacobian_scaling_alpha, Symbol("linsolve_identity!!"),
                :linsolve_workspace, :jacobian_initialized_preinverted,
                :make_identity!!,
                :maybe_unaliased, :maybe_unwrap_prob_for_enzyme, :reinit!, :reinit_common!,
                :reinit_self!, :reset!, :reset_timer!, :restructure, :run_initialization!,
                :safe_similar, :safe_vec, :solve!, :standardize_forwarddiff_tag, :step!,
                :store_inverse_jacobian, :stores_full_jacobian, :unwrap_val,
            ),
        ),
        # Still non-public in their owning packages. @SciMLMessage / AbstractVerbosityPreset
        # / None dropped: now imported directly from their owner SciMLLogging (public there).
        #   NonlinearSolveBase: @static_timeit,
        #     AbstractApproximateJacobianStructure, AbstractApproximateJacobianUpdateRule,
        #     AbstractApproximateJacobianUpdateRuleCache, AbstractDescentDirection,
        #     AbstractJacobianCache, AbstractJacobianInitialization,
        #     AbstractNonlinearSolveAlgorithm, AbstractNonlinearSolveCache,
        #     AbstractResetCondition, AbstractResetConditionCache, Utils, get_timer_output,
        #     update_trace!
        #   SciMLBase: NoSpecialize;  StaticArraysCore: StaticArray
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@static_timeit"),
                :AbstractApproximateJacobianStructure, :AbstractApproximateJacobianUpdateRule,
                :AbstractApproximateJacobianUpdateRuleCache, :AbstractDescentDirection,
                :AbstractJacobianCache, :AbstractJacobianInitialization,
                :AbstractNonlinearSolveAlgorithm, :AbstractNonlinearSolveCache,
                :AbstractResetCondition, :AbstractResetConditionCache, :NoSpecialize,
                :StaticArray, :Utils, :get_timer_output, :update_trace!,
            ),
        ),
    ),
)
