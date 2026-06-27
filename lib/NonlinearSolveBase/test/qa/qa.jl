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
        # Still non-public in their owning packages after the make-public round (the
        # ForwardDiff / SparseArrays extensions reach those packages' internals +
        # NonlinearSolveBase's own internal API; the main module reaches Base/Core/
        # LinearAlgebra/FunctionWrappers internals):
        #   SciMLBase: ChainRulesOriginator, DAEInitializationAlgorithm, EnzymeOriginator,
        #     NonNumberEltypeError, OverrideInitData, Void, __init, __solve, get_root_indp,
        #     has_colorvec, has_initialization_data, isdualtype,
        #     set_mooncakeoriginator_if_mooncake, specialization
        #   ForwardDiff: Dual, Partials, Tag, can_dual, derivative, gradient, jacobian,
        #     partials, pickchunksize, value
        #   Base: add_sum;  Core(.Compiler): Compiler, return_type;  LinearAlgebra: inv!
        #   FunctionWrappers: FunctionWrapper
        #   NonlinearSolveBase(.Utils/.InternalAPI): additional_incompatible_backend_check,
        #     condition_number, get_raw_f, get_u, make_sparse, maybe_pinv!!_workspace,
        #     maybe_symmetric, nlls_generate_vjp_function, nodual_value,
        #     nonlinearsolve_∂f_∂p, nonlinearsolve_∂f_∂u, reinit!, restructure, safe_reshape,
        #     safe_similar, sparse_or_structured_prototype, value
        all_qualified_accesses_are_public = (;
            ignore = (
                :ChainRulesOriginator, :DAEInitializationAlgorithm, :EnzymeOriginator,
                :NonNumberEltypeError, :OverrideInitData, :Void, :__init, :__solve,
                :get_root_indp, :has_colorvec, :has_initialization_data, :isdualtype,
                :set_mooncakeoriginator_if_mooncake, :specialization,
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
        # Still non-public in their owning packages after the make-public round:
        #   SciMLBase: AbstractODEIntegrator, ImmutableNonlinearProblem, KeywordArgError,
        #     NoDefaultAlgorithmError, NonSolverError, __init, __solve,
        #     _concrete_solve_adjoint, _concrete_solve_forward, checkkwargs, extract_alg,
        #     get_concrete_p, get_concrete_u0, get_root_indp, has_kwargs, promote_u0, wrap_sol
        #   ForwardDiff: Dual, pickchunksize;  SciMLOperators: AbstractSciMLOperator
        #   SparseArrays: AbstractSparseMatrixCSC;  StaticArraysCore: StaticArray
        #   NonlinearSolveBase (own internal): ImmutableNonlinearProblem,
        #     NonlinearSolveForwardDiffCache, NonlinearSolveTag, Utils, is_fw_wrapped,
        #     standardize_forwarddiff_tag, wrapfun_iip
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractODEIntegrator, :ImmutableNonlinearProblem, :KeywordArgError,
                :NoDefaultAlgorithmError, :NonSolverError, :__init, :__solve,
                :_concrete_solve_adjoint, :_concrete_solve_forward, :checkkwargs,
                :extract_alg, :get_concrete_p, :get_concrete_u0, :get_root_indp, :has_kwargs,
                :promote_u0, :wrap_sol, :Dual, :pickchunksize, :AbstractSciMLOperator,
                :AbstractSparseMatrixCSC, :StaticArray, :NonlinearSolveForwardDiffCache,
                :NonlinearSolveTag, :Utils, :is_fw_wrapped, :standardize_forwarddiff_tag,
                :wrapfun_iip,
            ),
        ),
    ),
)
