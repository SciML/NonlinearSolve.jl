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
        # Non-public names qualified-accessed from their owning packages (ForwardDiff ext):
        #   ForwardDiff: Dual, Partials, Tag, can_dual, derivative, gradient, jacobian,
        #     partials, pickchunksize, value
        #   SciMLBase: Void, __init, __solve, build_solution
        #   NonlinearSolveBase(.Utils/.InternalAPI, own internal):
        #     additional_incompatible_backend_check, get_raw_f, get_u,
        #     nlls_generate_vjp_function, nodual_value, nonlinearsolve_∂f_∂p,
        #     nonlinearsolve_∂f_∂u, reinit!, restructure, safe_reshape, safe_similar, value
        #   CommonSolve: solve!
        all_qualified_accesses_are_public = (;
            ignore = (
                :Dual, :Partials, :Tag, :Void, :__init, :__solve,
                :additional_incompatible_backend_check, :build_solution, :can_dual,
                :derivative, :get_raw_f, :get_u, :gradient, :jacobian,
                :nlls_generate_vjp_function, :nodual_value,
                Symbol("nonlinearsolve_∂f_∂p"), Symbol("nonlinearsolve_∂f_∂u"),
                :partials, :pickchunksize, :reinit!, :restructure, :safe_reshape,
                :safe_similar, :solve!, :value,
            ),
        ),
        # Non-public names explicitly imported from their owning packages (ForwardDiff ext):
        #   NonlinearSolveBase (own internal): ImmutableNonlinearProblem,
        #     NonlinearSolveForwardDiffCache, NonlinearSolveTag, Utils, is_fw_wrapped,
        #     standardize_forwarddiff_tag, wrapfun_iip
        #   SciMLBase: AbstractNonlinearProblem
        #   ForwardDiff: Dual, pickchunksize
        #   CommonSolve: init, solve, solve!
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearProblem, :Dual, :ImmutableNonlinearProblem,
                :NonlinearSolveForwardDiffCache, :NonlinearSolveTag, :Utils, :init,
                :is_fw_wrapped, :pickchunksize, :solve, :solve!,
                :standardize_forwarddiff_tag, :wrapfun_iip,
            ),
        ),
    ),
)
