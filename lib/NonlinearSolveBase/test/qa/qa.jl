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
        # SciMLLogging preset names reached only through the @verbosity_specifier macro
        # expansion (None/Minimal/Standard/Detailed/All as bare constructors in
        # src/verbosity.jl + AbstractVerbositySpecifier/MessageLevel in the docstring +
        # supertype). ExplicitImports cannot see through the macro, so they look stale;
        # removing the imports would break precompile (the macro body resolves them in
        # NonlinearSolveBase's scope).
        no_stale_explicit_imports = (;
            ignore = (
                :AbstractVerbositySpecifier, :All, :MessageLevel, :Minimal, :Standard,
            ),
        ),
        # Still non-public in their owning packages (the ForwardDiff / SparseArrays
        # extensions reach those packages' internals + NonlinearSolveBase's own internal
        # API; the main module reaches Base/Core/LinearAlgebra/FunctionWrappers internals).
        # __init / __solve / has_initialization_data dropped here: now public in SciMLBase.
        #   SciMLBase: ChainRulesOriginator, DAEInitializationAlgorithm, EnzymeOriginator,
        #     NonNumberEltypeError, OverrideInitData, Void, get_root_indp, has_colorvec,
        #     isdualtype, set_mooncakeoriginator_if_mooncake, specialization
        #   ForwardDiff: Dual, Partials, Tag, can_dual, derivative, gradient, jacobian,
        #     partials, pickchunksize, value
        #   Base: add_sum;  Core(.Compiler): Compiler, return_type;  LinearAlgebra: inv!
        #   FunctionWrappers: FunctionWrapper
        #   NonlinearSolveBase(.Utils/.InternalAPI): additional_incompatible_backend_check,
        #     condition_number, get_raw_f, get_u, make_sparse, maybe_pinv!!_workspace,
        #     maybe_symmetric, nlls_generate_vjp_function, nodual_value,
        #     nonlinearsolve_∂f_∂p, nonlinearsolve_∂f_∂u, reinit!, restructure, safe_reshape,
        #     safe_similar, sparse_or_structured_prototype
        all_qualified_accesses_are_public = (;
            ignore = (
                :ChainRulesOriginator, :DAEInitializationAlgorithm, :EnzymeOriginator,
                :NonNumberEltypeError, :OverrideInitData, :Void,
                :get_root_indp, :has_colorvec, :isdualtype,
                :set_mooncakeoriginator_if_mooncake, :specialization,
                :Dual, :Partials, :Tag, :can_dual, :derivative, :gradient, :jacobian,
                :partials, :pickchunksize, :value, :add_sum, :Compiler, :return_type, :inv!,
                :FunctionWrapper, :additional_incompatible_backend_check, :condition_number,
                :get_raw_f, :get_u, :make_sparse, Symbol("maybe_pinv!!_workspace"),
                :maybe_symmetric, :nlls_generate_vjp_function, :nodual_value,
                Symbol("nonlinearsolve_∂f_∂p"), Symbol("nonlinearsolve_∂f_∂u"),
                :reinit!, :restructure, :safe_reshape, :safe_similar,
                :sparse_or_structured_prototype,
            ),
        ),
        # Still non-public in their owning packages. AbstractODEIntegrator / __init /
        # __solve dropped here: now public in SciMLBase.
        #   SciMLBase: ImmutableNonlinearProblem, KeywordArgError, NoDefaultAlgorithmError,
        #     NonSolverError, _concrete_solve_adjoint, _concrete_solve_forward, checkkwargs,
        #     extract_alg, get_concrete_p, get_concrete_u0, get_root_indp, has_kwargs,
        #     promote_u0, wrap_sol
        #   ForwardDiff: Dual, pickchunksize;  SciMLOperators: AbstractSciMLOperator
        #   SparseArrays: AbstractSparseMatrixCSC;  StaticArraysCore: StaticArray
        #   NonlinearSolveBase (own internal): NonlinearSolveForwardDiffCache,
        #     NonlinearSolveTag, Utils, is_fw_wrapped, standardize_forwarddiff_tag,
        #     wrapfun_iip
        all_explicit_imports_are_public = (;
            ignore = (
                :ImmutableNonlinearProblem, :KeywordArgError,
                :NoDefaultAlgorithmError, :NonSolverError,
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
