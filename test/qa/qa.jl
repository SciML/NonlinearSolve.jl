using SciMLTesting, NonlinearSolve, SimpleNonlinearSolve, SciMLBase, Aqua, Test
import ForwardDiff, JET, NonlinearSolveBase
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
    ei_kwargs = (;
        # NonDifferentiable is owned by NLSolversBase and re-exported through NLsolve
        # (where the NLsolve extension imports it from).
        all_explicit_imports_via_owners = (; ignore = (:NonDifferentiable,)),
        # Still non-public in their owning packages, across the main module and the solver
        # extensions. AbstractSteadyStateProblem / __init / __solve dropped: now public in
        # SciMLBase.
        #   NonlinearSolveBase(.Utils): Utils, evaluate_f, nodual_value,
        #     safe_vec, NonlinearSolveForwardDiffCache
        #   ForwardDiff: partials;  LeastSquaresOptim: Cholesky, LSMR, QR
        all_qualified_accesses_are_public = (;
            ignore = (
                :Utils, :evaluate_f, :nodual_value, :safe_vec,
                :NonlinearSolveForwardDiffCache,
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

@testset "JET type stability" begin
    FDExt = Base.get_extension(NonlinearSolveBase, :NonlinearSolveBaseForwardDiffExt)
    # Count only findings in frames owned by this repository; SciMLBase, ForwardDiff
    # and Base internals carry pre-existing possible-error reports that are not ours.
    targetmods = (NonlinearSolveBase, FDExt)
    jet_f!(du, u, p) = (du .= u .^ 3 .+ u .- p.tunable)
    fn = NonlinearFunction(jet_f!)
    DualT = typeof(ForwardDiff.Dual{:jettest}(3.0, 1.0))
    prob = NonlinearProblem(fn, [1.0], (; tunable = [3.0]))
    prob_dualp = NonlinearProblem(
        fn, [1.0], (; tunable = [ForwardDiff.Dual{:jettest}(3.0, 1.0)])
    )
    prob_dualu0 = NonlinearProblem(
        fn, [ForwardDiff.Dual{:jettest}(1.0, 1.0)], (; tunable = [3.0])
    )
    prob_flatp = NonlinearProblem(fn, [1.0], [ForwardDiff.Dual{:jettest}(3.0, 1.0)])

    # The ForwardDiff interception hooks must not add dispatch overhead on the
    # non-dual fast path.
    JET.test_opt(
        NonlinearSolveBase.InternalAPI.forwarddiff_solve,
        Tuple{typeof(prob), typeof(Broyden())},
    )
    JET.test_opt(
        NonlinearSolveBase.InternalAPI.forwarddiff_init,
        Tuple{typeof(prob), typeof(Broyden())},
    )

    # No possible-errors in our frames along the dual entry points.
    JET.test_call(() -> solve(prob_dualp, Broyden()); target_modules = targetmods)
    JET.test_call(() -> solve(prob_dualu0, Broyden()); target_modules = targetmods)
    JET.test_call(() -> solve(prob_flatp, Broyden()); target_modules = targetmods)
    JET.test_call(() -> init(prob_dualp, Broyden()); target_modules = targetmods)
    JET.test_call(
        () -> solve!(init(prob_dualp, Broyden())); target_modules = targetmods
    )
    JET.test_call(
        () -> SciMLBase.reinit!(
            init(prob_dualp, Broyden()), [1.0]; p = (; tunable = [4.0])
        ); target_modules = targetmods
    )

    # The strip/rebuild leaf helpers are fully inferred on concrete leaves.
    JET.test_opt(FDExt._dual_view, Tuple{DualT})
    JET.test_opt(FDExt._dual_view, Tuple{Vector{DualT}})
    JET.test_opt(FDExt._zero_partials, Tuple{Vector{Float64}, Vector{DualT}})
    JET.test_opt(
        FDExt._forwarddiff_combine_partials,
        Tuple{Matrix{Float64}, Vector{DualT}, Vector{Float64}},
    )
    JET.test_opt(
        NonlinearSolveBase.nonlinearsolve_dual_solution,
        Tuple{Vector{Float64}, Vector{ForwardDiff.Partials{1, Float64}}, Vector{DualT}},
    )

    # Every ForwardDiff cache field stays concrete through init and reinit!.
    cache = init(prob_dualp, Broyden())
    @test all(isconcretetype, fieldtypes(typeof(cache)))
    cache_u0 = init(prob_dualu0, Broyden())
    @test all(isconcretetype, fieldtypes(typeof(cache_u0)))
    @test cache_u0.u0duals isa Vector{DualT} && !isempty(cache_u0.u0duals)
    SciMLBase.reinit!(cache, [1.0]; p = (; tunable = [4.0]))
    @test all(isconcretetype, fieldtypes(typeof(cache)))
    @test isempty(cache.u0duals)
end
