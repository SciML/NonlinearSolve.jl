using SciMLTesting, NonlinearSolveBase, Test
using NonlinearSolveBase: AbstractNonlinearProblem, NonlinearProblem, SciMLBase
import ForwardDiff, SparseArrays

run_qa(
    NonlinearSolveBase;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:TimerOutputs]),
        piracies = (;
            treat_as_own = [
                AbstractNonlinearProblem, NonlinearProblem, SciMLBase.HomotopyProblem,
            ],
        ),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # SciMLLogging types reached only through the @verbosity_specifier macro;
        # ExplicitImports cannot track macro usage, so they look stale.
        no_stale_explicit_imports = (;
            ignore = (
                :MessageLevel, :AbstractVerbositySpecifier, :All, :Detailed, :Minimal,
                :None, :Standard, :SciMLLogging,
            ),
        ),
        # ImmutableNonlinearProblem is owned by SciMLBase and re-exported through
        # NonlinearSolveBase (where the ForwardDiff extension imports it from).
        all_explicit_imports_via_owners = (; ignore = (:ImmutableNonlinearProblem,)),
        # Still non-public in their owning packages (the public-API round covered only
        # SciMLBase/CommonSolve/ArrayInterface; not ForwardDiff or NonlinearSolveBase's own
        # internal API), across the main module and the ForwardDiff / SparseArrays /
        # LinearSolve / SparseMatrixColorings extensions:
        #   SciMLBase: ChainRulesOriginator, DAEInitializationAlgorithm, EnzymeOriginator,
        #     NoInit, NonNumberEltypeError, OverrideInit, OverrideInitData, Void, __init,
        #     __solve, allows_late_binding_tstops, allowsbounds, get_initial_values,
        #     get_root_indp, has_colorvec, has_initialization_data, isdualtype,
        #     set_mooncakeoriginator_if_mooncake, specialization
        #   ForwardDiff: Dual, Partials, Tag, can_dual, derivative, gradient, jacobian,
        #     partials, pickchunksize, value
        #   Base: add_sum;  Core(.Compiler): Compiler, return_type;  LinearAlgebra: inv!
        #   FunctionWrappers: FunctionWrapper
        #   NonlinearSolveBase(.Utils/.InternalAPI, own internal):
        #     additional_incompatible_backend_check, condition_number, get_raw_f, get_u,
        #     make_sparse, maybe_pinv!!_workspace, maybe_symmetric, nlls_generate_vjp_function,
        #     nodual_value, nonlinearsolve_∂f_∂p, nonlinearsolve_∂f_∂u, reinit!, restructure,
        #     safe_reshape, safe_similar, sparse_or_structured_prototype
        all_qualified_accesses_are_public = (;
            ignore = (
                :ChainRulesOriginator, :DAEInitializationAlgorithm, :EnzymeOriginator,
                :NoInit, :NonNumberEltypeError, :OverrideInit, :OverrideInitData, :Void,
                :__init, :__solve, :allows_late_binding_tstops, :allowsbounds,
                :get_initial_values, :get_root_indp, :has_colorvec, :has_initialization_data,
                :isdualtype, :set_mooncakeoriginator_if_mooncake, :specialization,
                :Dual, :Partials, :Tag, :can_dual, :derivative, :gradient, :jacobian,
                :partials, :pickchunksize, :value, :add_sum, :Compiler, :return_type, :inv!,
                :FunctionWrapper, :additional_incompatible_backend_check, :condition_number,
                :get_raw_f, :get_u, :make_sparse, :maybe_pinv!!_workspace, :maybe_symmetric,
                :nlls_generate_vjp_function, :nodual_value,
                Symbol("nonlinearsolve_∂f_∂p"), Symbol("nonlinearsolve_∂f_∂u"),
                :reinit!, :restructure, :safe_reshape, :safe_similar,
                :sparse_or_structured_prototype,
            ),
        ),
        # Still non-public in their owning packages, across the main module and the
        # ForwardDiff / SparseArrays / LinearSolve extensions:
        #   SciMLBase: AbstractNonlinearAlgorithm, AbstractODEIntegrator,
        #     ImmutableNonlinearProblem, KeywordArgError, NoDefaultAlgorithmError,
        #     NonSolverError, NonlinearAliasSpecifier, __init, __solve,
        #     _concrete_solve_adjoint, _concrete_solve_forward, checkkwargs, extract_alg,
        #     get_concrete_p, get_concrete_u0, get_root_indp, has_kwargs, promote_u0, wrap_sol
        #   SciMLOperators: AbstractSciMLOperator;  StaticArraysCore: StaticArray
        #   SparseArrays: AbstractSparseMatrixCSC;  ForwardDiff: Dual, pickchunksize
        #   NonlinearSolveBase (own internal): NonlinearSolveForwardDiffCache,
        #     NonlinearSolveTag, Utils, is_fw_wrapped, standardize_forwarddiff_tag, wrapfun_iip
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearAlgorithm, :AbstractODEIntegrator,
                :ImmutableNonlinearProblem, :KeywordArgError, :NoDefaultAlgorithmError,
                :NonSolverError, :NonlinearAliasSpecifier, :__init, :__solve,
                :_concrete_solve_adjoint, :_concrete_solve_forward, :checkkwargs,
                :extract_alg, :get_concrete_p, :get_concrete_u0, :get_root_indp, :has_kwargs,
                :promote_u0, :wrap_sol, :AbstractSciMLOperator, :StaticArray,
                :AbstractSparseMatrixCSC, :Dual, :pickchunksize,
                :NonlinearSolveForwardDiffCache, :NonlinearSolveTag, :Utils, :is_fw_wrapped,
                :standardize_forwarddiff_tag, :wrapfun_iip,
            ),
        ),
    ),
)
