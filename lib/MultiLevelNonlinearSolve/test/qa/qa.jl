using SciMLTesting, MultiLevelNonlinearSolve, Test

const NONLINEARSOLVE_DOCS_SRC = joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src")

const MLN_EXTERNAL_REEXPORTS = union(
    public_api_names(MultiLevelNonlinearSolve.SciMLBase),
    public_api_names(MultiLevelNonlinearSolve.NonlinearSolveBase),
    public_api_names(MultiLevelNonlinearSolve.NonlinearSolveFirstOrder),
    (:SciMLBase, :NonlinearSolveBase, :NonlinearSolveFirstOrder),
)

run_qa(
    MultiLevelNonlinearSolve;
    explicit_imports = true,
    reexports_allow = MLN_EXTERNAL_REEXPORTS,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = NONLINEARSOLVE_DOCS_SRC,
        ignore = MLN_EXTERNAL_REEXPORTS,
        rendered_ignore = MLN_EXTERNAL_REEXPORTS,
    ),
    ei_kwargs = (;
        # Still non-public in their owning packages (NonlinearSolveBase's own internal API;
        # the sublibrary builds on it by design).
        all_qualified_accesses_are_public = (;
            ignore = (
                :NonlinearSolveDefaultInit, :apply_postcondition!!, :evaluate_f!, :get_fu,
                :get_postcondition, :get_tolerance, :get_u, :init_nonlinearsolve_trace,
                :maybe_unaliased, :reinit!, :reset!, :reset_timer!, :run_initialization!,
                :safe_getproperty, :safe_similar, :set_du!, :step!,
                :supports_postcondition, :update_from_termination_cache!, :update_trace!,
                :AbstractNonlinearTerminationMode,
                :AbstractSafeBestNonlinearTerminationMode, :L2_NORM, :Linf_NORM,
                :apply_norm, :normalize_verbosity, :needs_bounds_transform,
                # `structdiff` filters the forwarded solve keywords; the public alternatives
                # are all generator-based and opaque to inference at that call site.
                :structdiff,
                # `Base.RefValue` is the concrete cell type the local-tolerance and
                # last-assembled-S fields are declared as; `Ref` itself is abstract.
                :RefValue,
                # Implementing a LinearSolve algorithm means implementing its interface,
                # every hook of which is internal to LinearSolve.
                :AbstractFactorization, :LinearCache, :default_alias_A, :default_alias_b,
                :init_cacheval, :needs_concrete_A, :update_tolerances!,
                :update_tolerances_internal!,
                # The nonlinear-preconditioning composition hook, likewise internal.
                :compose_precondition,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@static_timeit"), :AbstractNonlinearSolveAlgorithm,
                :AbstractNonlinearSolveCache, :InternalAPI, :Utils, :get_timer_output,
            ),
        ),
    ),
)
