using SciMLTesting, NonlinearSolve, SimpleNonlinearSolve, SciMLBase, Aqua, Test
# Load the wrapper packages so NonlinearSolve's solver extensions are present for
# ExplicitImports to analyze. An extension module only exists once every one of its
# triggers is loaded, and ExplicitImports skips extensions that do not exist, so a
# missing trigger here silently drops that extension from every check below.
# LineSearches is the second trigger of NonlinearSolveNLsolveExt: NLsolve happens to
# depend on it today, but list it explicitly so the coverage does not hinge on that.
# NonlinearSolvePETScExt stays unscanned: PETSc/MPI need an external MPI + PETSc
# installation, which is not something the QA environment can resolve.
using ADTypes
import FastLevenbergMarquardt, FixedPointAcceleration, LeastSquaresOptim, LineSearches,
    MINPACK, NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping, Sundials

# ExplicitImports silently skips an extension that fails to load, so assert the extension
# modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    for ext in (
            :NonlinearSolveFastLevenbergMarquardtExt,
            :NonlinearSolveFixedPointAccelerationExt,
            :NonlinearSolveLeastSquaresOptimExt,
            :NonlinearSolveMINPACKExt,
            :NonlinearSolveNLSolversExt,
            :NonlinearSolveNLsolveExt,
            :NonlinearSolveSIAMFANLEquationsExt,
            :NonlinearSolveSpeedMappingExt,
            :NonlinearSolveSundialsExt,
        )
        @test Base.get_extension(NonlinearSolve, ext) !== nothing
    end
end

const NONLINEARSOLVE_EXTERNAL_REEXPORTS = union(
    public_api_names(NonlinearSolve.ADTypes),
    public_api_names(NonlinearSolve.SciMLBase),
    public_api_names(NonlinearSolve.LineSearch),
    public_api_names(NonlinearSolve.LinearSolve),
    (:ADTypes, :SciMLBase, :LineSearch, :LinearSolve),
)

# NonlinearSolve is a facade: it deliberately re-exports the whole solver stack, so every
# public name of a sublibrary (and the sublibrary module names themselves) is an intended
# public re-export rather than an accidental one. This is the allow-list for
# `check_reexports`; the external re-exports above are intended in the same way.
const NONLINEARSOLVE_SUBLIBRARY_REEXPORTS = union(
    public_api_names(NonlinearSolve.NonlinearSolveBase),
    public_api_names(NonlinearSolve.NonlinearSolveFirstOrder),
    public_api_names(NonlinearSolve.NonlinearSolveSpectralMethods),
    public_api_names(NonlinearSolve.NonlinearSolveQuasiNewton),
    public_api_names(NonlinearSolve.SimpleNonlinearSolve),
    public_api_names(NonlinearSolve.BracketingNonlinearSolve),
    (
        :NonlinearSolveBase, :NonlinearSolveFirstOrder, :NonlinearSolveSpectralMethods,
        :NonlinearSolveQuasiNewton, :SimpleNonlinearSolve, :BracketingNonlinearSolve,
    ),
)

const NONLINEARSOLVE_ALLOWED_REEXPORTS = union(
    NONLINEARSOLVE_EXTERNAL_REEXPORTS, NONLINEARSOLVE_SUBLIBRARY_REEXPORTS
)

run_qa(
    NonlinearSolve;
    explicit_imports = true,
    reexports_allow = NONLINEARSOLVE_ALLOWED_REEXPORTS,
    aqua_kwargs = (;
        # stale_deps / deps_compat are checked on the SimpleNonlinearSolve facade
        # below (with the SciMLJacobianOperators ignore); persistent_tasks stays off
        # for the umbrella package.
        stale_deps = false,
        deps_compat = false,
        persistent_tasks = false,
        ambiguities = (; recursive = false),
        piracies = (;
            treat_as_own = [
                NonlinearProblem, NonlinearLeastSquaresProblem,
                SciMLBase.AbstractNonlinearProblem,
                # `initialization_alg` is dispatched here for the continuation problem
                # type too, alongside the `AbstractNonlinearProblem` method above.
                SciMLBase.HomotopyProblem,
                SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm,
            ],
        ),
    ),
    api_docs_kwargs = (;
        rendered = true,
        ignore = NONLINEARSOLVE_EXTERNAL_REEXPORTS,
        rendered_ignore = NONLINEARSOLVE_EXTERNAL_REEXPORTS,
    ),
    ei_kwargs = (;
        # NonDifferentiable is owned by NLSolversBase and re-exported through NLsolve
        # (where the NLsolve extension imports it from).
        all_explicit_imports_via_owners = (; ignore = (:NonDifferentiable,)),
        # Still non-public in their owning packages, across the main module and the solver
        # extensions. AbstractSteadyStateProblem / __init / __solve dropped: now public in
        # SciMLBase.
        #   NonlinearSolveBase(.Utils): Utils, evaluate_f, initialization_alg, nodual_value,
        #     safe_vec
        #   ForwardDiff: partials;  LeastSquaresOptim: Cholesky, LSMR, QR
        all_qualified_accesses_are_public = (;
            ignore = (
                :Utils, :evaluate_f, :initialization_alg, :nodual_value, :safe_vec,
                :partials, :Cholesky, :LSMR, :QR,
            ),
        ),
        # Still non-public in their owning packages after the make-public round:
        #   NonlinearSolveBase: AbstractNonlinearSolveAlgorithm, Utils, get_raw_f,
        #     is_fw_wrapped
        #   ForwardDiff: Dual;  StaticArraysCore: StaticArray
        #   NonlinearSolveFirstOrder: RUS;  NLsolve (re-export, owner NLSolversBase):
        #     NonDifferentiable
        #   NonlinearSolve (own internal): DualNonlinearProblem, the dispatch alias the
        #     Sundials extension needs to attach its ForwardDiff-over-KINSOL methods to.
        #     There is no public spelling of it and it is not part of the user-facing API.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearSolveAlgorithm, :Utils, :get_raw_f, :is_fw_wrapped, :Dual,
                :StaticArray, :RUS, :NonDifferentiable, :DualNonlinearProblem,
            ),
        ),
    ),
)

# stale_deps / deps_compat are validated via the SimpleNonlinearSolve facade, which
# carries the SciMLJacobianOperators weak-dep ignore.
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
