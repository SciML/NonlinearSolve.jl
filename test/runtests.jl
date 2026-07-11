using Pkg
using SafeTestsets, Test, InteractiveUtils
using SciMLTesting

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch. The root CI (grouped-tests.yml@v1) and SublibraryCI both set
# NONLINEARSOLVE_TEST_GROUP; a bare local run may set GROUP instead. Read either so
# the same runtests.jl works as the root test entry and the sublibrary dispatcher.
# run_tests reads NONLINEARSOLVE_TEST_GROUP below, so seed it from GROUP when only
# GROUP is set (preserving the previous `get(ENV, "NONLINEARSOLVE_TEST_GROUP",
# get(ENV, "GROUP", "All"))` precedence).
if !haskey(ENV, "NONLINEARSOLVE_TEST_GROUP") && haskey(ENV, "GROUP")
    ENV["NONLINEARSOLVE_TEST_GROUP"] = ENV["GROUP"]
end

const GROUP = current_group(; env = "NONLINEARSOLVE_TEST_GROUP")
const LIB_DIR = joinpath(dirname(@__DIR__), "lib")

@info "Running tests for group: $(GROUP)"

if GROUP == "Trim"
    # Trimming was introduced in Julia 1.12; runs in its own environment. Modeled as
    # a self-contained group rather than a run_tests group so the version gate and
    # the bespoke (root-not-developed) trim env activation are preserved verbatim.
    if VERSION >= v"1.12.0-rc1"
        Pkg.activate(joinpath(@__DIR__, "trim"))
        Pkg.instantiate()
        include("trim/runtests.jl")
    end
else
    run_tests(;
        env = "NONLINEARSOLVE_TEST_GROUP",
        # --- Base-env groups (no extra deps; part of the "All" run) ---
        core = function ()
            @time @safetestset "NLLS Analytic Jacobian" include("Core/core_tests__item1.jl")
            @time @safetestset "PolyAlgorithm Type Inference" include("Core/core_tests__item3.jl")
            @time @safetestset "PolyAlgorithm Aliasing" include("Core/core_tests__item5.jl")
            @time @safetestset "BigFloat Support" include("Core/core_tests__item7.jl")
            @time @safetestset "Singular Exception: Issue #153" include("Core/core_tests__item8.jl")
            @time @safetestset "Simple Scalar Problem: Issue #187" include("Core/core_tests__item9.jl")
            @time @safetestset "No AD" include("Core/core_tests__item11.jl")
            @time @safetestset "Infeasible" include("Core/core_tests__item12.jl")
            @time @safetestset "NoInit Caching" include("Core/core_tests__item13.jl")
            @time @safetestset "Singular Systems -- Auto Linear Solve Switching" include("Core/core_tests__item16.jl")
            @time @safetestset "No PolyesterForwardDiff for SArray" include("Core/core_tests__item17.jl")
            @time @safetestset "NonlinearLeastSquares ReturnCode" include("Core/core_tests__item18.jl")
            @time @safetestset "NonNumberEltype error" include("Core/core_tests__item20.jl")
            @time @safetestset "LinearSolve Preconditioner Interface" include("Core/core_tests__item21.jl")
            @time @safetestset "23 Test Problems: NewtonRaphson" include("Core/23_test_problems_tests__item2.jl")
            @time @safetestset "23 Test Problems: Halley" include("Core/23_test_problems_tests__item3.jl")
            @time @safetestset "23 Test Problems: TrustRegion" include("Core/23_test_problems_tests__item4.jl")
            @time @safetestset "23 Test Problems: LevenbergMarquardt" include("Core/23_test_problems_tests__item5.jl")
            @time @safetestset "23 Test Problems: DFSane" include("Core/23_test_problems_tests__item6.jl")
            @time @safetestset "23 Test Problems: Broyden" include("Core/23_test_problems_tests__item7.jl")
            @time @safetestset "23 Test Problems: Klement" include("Core/23_test_problems_tests__item8.jl")
            @time @safetestset "23 Test Problems: PseudoTransient" include("Core/23_test_problems_tests__item9.jl")
            @time @safetestset "Default Algorithm for AbstractSteadyStateProblem" include("Core/default_alg_tests__item1.jl")
            @time @safetestset "NLLS Hessian SciML/NonlinearSolve.jl#445" include("Core/forward_ad_tests__item2.jl")
            @time @safetestset "reinit! on ForwardDiff cache SciML/NonlinearSolve.jl#391" include("Core/forward_ad_tests__item3.jl")
            @time @safetestset "reinit! preserves parameters (LinearSolveParameters type stability)" include("Core/reinit_tests__item1.jl")
            @time @safetestset "Correct Best Solution: #565" include("Core/issue_tests__item1.jl")
            @time @safetestset "Polyalgorithm Fallback Path: CurveFit.jl#76" include("Core/issue_tests__item2.jl")
            @time @safetestset "Polyalgorithm Cache solve!: Issue #779" include("Core/issue_tests__item3.jl")
            @time @safetestset "Bounds: NonlinearLeastSquaresProblem" include("Core/bounds_tests__item1.jl")
            @time @safetestset "Bounds: one-sided" include("Core/bounds_tests__item2.jl")
            @time @safetestset "Bounds: polyalgorithm and quasi-Newton algorithms" include("Core/bounds_tests__item4.jl")
            @time @safetestset "HomotopySweep construction + defaults" include("Core/homotopy_sweep_tests__item1.jl")
            @time @safetestset "HomotopySweep happy path (oop, λ as separate argument)" include("Core/homotopy_sweep_tests__item2.jl")
            @time @safetestset "HomotopySweep in-place residual f(du, u, p, λ)" include("Core/homotopy_sweep_tests__item3.jl")
            @time @safetestset "HomotopySweep with structured (NamedTuple) parameters" include("Core/homotopy_sweep_tests__item4.jl")
            @time @safetestset "HomotopySweep rescues an out-of-basin guess (no MTK)" include("Core/homotopy_sweep_tests__item5.jl")
            @time @safetestset "HomotopySweep reports a failure retcode when it cannot finish" include("Core/homotopy_sweep_tests__item6.jl")
            @time @safetestset "HomotopySweep returns last converged iterate even with aliasing" include("Core/homotopy_sweep_tests__item7.jl")
            @time @safetestset "HomotopySweep honors HomotopyProblem-level solver kwargs" include("Core/homotopy_sweep_tests__item8.jl")
            @time @safetestset "HomotopySweep adaptive=false takes fixed steps and fails fast" include("Core/homotopy_sweep_tests__item9.jl")
            @time @safetestset "HomotopySweep stalls (does not hang) when dλ underflows eps(λ)" include("Core/homotopy_sweep_tests__item10.jl")
            @time @safetestset "HomotopySweep handles a decreasing λspan" include("Core/homotopy_sweep_tests__item11.jl")
            @time @safetestset "HomotopySweep stays in Float32 (no promotion)" include("Core/homotopy_sweep_tests__item12.jl")
            @time @safetestset "HomotopySweep inner solver is composable" include("Core/homotopy_sweep_tests__item13.jl")
            @time @safetestset "HomotopyProblem defaults to HomotopySweep when alg is nothing" include("Core/homotopy_sweep_tests__item14.jl")
            @time @safetestset "HomotopySweep anchors at λspan[1] (right branch, not wrong root)" include("Core/homotopy_sweep_tests__item15.jl")
            @time @safetestset "HomotopySweep fails fast when the λspan[1] anchor is unsolvable" include("Core/homotopy_sweep_tests__item16.jl")
            @time @safetestset "HomotopySweep solves a zero-width λspan exactly once" include("Core/homotopy_sweep_tests__item17.jl")
            @time @safetestset "HomotopySweep step-control kwargs validation + defaults" include("Core/homotopy_sweep_tests__item18.jl")
            @time @safetestset "HomotopySweep expands the step after consecutive successes" include("Core/homotopy_sweep_tests__item19.jl")
            @time @safetestset "HomotopySweep secant predictor reduces corrector work" include("Core/homotopy_sweep_tests__item20.jl")
            @time @safetestset "HomotopySweep regrows the step after bisecting a hard region" include("Core/homotopy_sweep_tests__item21.jl")
            @time @safetestset "HomotopySweep solution field reads do not allocate through getproperty" include("Core/homotopy_sweep_tests__item22.jl")
            @time @safetestset "ArcLengthContinuation construction + defaults" include("Core/arclength_tests__item1.jl")
            @time @safetestset "ArcLengthContinuation happy path (fold-free, matches sweep)" include("Core/arclength_tests__item2.jl")
            @time @safetestset "ArcLengthContinuation rounds a fold (non-monotone λ)" include("Core/arclength_tests__item3.jl")
            @time @safetestset "ArcLengthContinuation Float32 / in-place / multi-dim" include("Core/arclength_tests__item4.jl")
            @time @safetestset "ArcLengthContinuation fails (no hang) on an unreachable target" include("Core/arclength_tests__item5.jl")
            @time @safetestset "ArcLengthContinuation tangent predictor" include("Core/arclength_tests__item6.jl")
            @time @safetestset "ArcLengthContinuation bordered tangent + θ-weighted metric" include("Core/arclength_tests__item7.jl")
            @time @safetestset "ArcLengthContinuation corrector cache per-step allocations" include("Core/arclength_tests__item8.jl")
            @time @safetestset "Homotopy sweeps consume jac/jac_prototype/sparsity/colorvec" include("Core/homotopy_jac_tests__item1.jl")
            @time @safetestset "ArcLengthContinuation consumes jac/jac_prototype/sparsity/colorvec" include("Core/arclength_jac_tests__item1.jl")
            @time @safetestset "Homotopy tracking_maxiters caps interior corrector work" include("Core/homotopy_effort_tests__item1.jl")
            @time @safetestset "Homotopy maxsteps guard + effort-band step control" include("Core/homotopy_effort_tests__item2.jl")
            @time @safetestset "Homotopy tracking_abstol loosens interior tracking only" include("Core/homotopy_tolerance_tests__item1.jl")
            @time @safetestset "PolyAlgorithm best-subalgorithm retention (reinit! retain_best)" include("Core/polyalg_retention_tests__item1.jl")
            @time @safetestset "Homotopy drivers retain the winning inner subalgorithm" include("Core/homotopy_retention_tests__item1.jl")
            @time @safetestset "HomotopyPolyAlgorithm stages, fallback, both-fail" include("Core/homotopy_polyalg_tests__item1.jl")
            return @time @safetestset "HomotopyPolyAlgorithm warm sweep→fallback handoff" include("Core/homotopy_polyalg_tests__item2.jl")
        end,
        groups = Dict(
            "PolyAlgorithms" => function ()
                @time @safetestset "Basic PolyAlgorithms" include("PolyAlgorithms/core_tests__item2.jl")
                @time @safetestset "PolyAlgorithms Autodiff" include("PolyAlgorithms/core_tests__item4.jl")
                @time @safetestset "Ensemble Nonlinear Problems" include("PolyAlgorithms/core_tests__item6.jl")
                @time @safetestset "Out-of-place Matrix Resizing" include("PolyAlgorithms/core_tests__item14.jl")
                @time @safetestset "Inplace Matrix Resizing" include("PolyAlgorithms/core_tests__item15.jl")
                @time @safetestset "Default Algorithm Singular Handling" include("PolyAlgorithms/core_tests__item19.jl")
                @time @safetestset "Default polyalgs are forward-mode only: Issue #837" include("PolyAlgorithms/core_tests__item20.jl")
                return @time @safetestset "23 Test Problems: PolyAlgorithms" include("PolyAlgorithms/23_test_problems_tests__item1.jl")
            end,
            "Verbosity" => function ()
                return @time @safetestset "Nonlinear Verbosity" include("Verbosity/verbosity_tests__item1.jl")
            end,
            # --- Dep-adding groups (isolated sub-envs; excluded from "All") ---
            "Downstream" => (;
                env = joinpath(@__DIR__, "Downstream"),
                body = function ()
                    @time @safetestset "Complex Valued Problems: Single-Shooting" include("Downstream/core_tests__item10.jl")
                    return @time @safetestset "Modeling Toolkit Cache Indexing" include("Downstream/mtk_cache_indexing_tests__item1.jl")
                end,
            ),
            "Bounds" => (;
                env = joinpath(@__DIR__, "Bounds"),
                body = function ()
                    return @time @safetestset "Bounds: nonlinear model" include("Bounds/bounds_tests__item3.jl")
                end,
            ),
            "Adjoint" => (;
                env = joinpath(@__DIR__, "Adjoint"),
                body = function ()
                    @time @safetestset "Adjoint Tests" include("Adjoint/adjoint_tests__item1.jl")
                    @time @safetestset "maybe_wrap_nonlinear_f skips wrapping inside Enzyme.autodiff (#939)" include("Adjoint/adjoint_tests__item2.jl")
                    return @time @safetestset "Enzyme reverse-mode over IIP NonlinearProblem (#939)" include("Adjoint/adjoint_tests__item3.jl")
                end,
            ),
            "Wrappers" => (;
                env = joinpath(@__DIR__, "Wrappers"),
                body = function ()
                    @time @safetestset "ForwardDiff.jl Integration" include("Wrappers/forward_ad_tests__item1.jl")
                    @time @safetestset "Simple Scalar Problem" include("Wrappers/fixedpoint_tests__item1.jl")
                    @time @safetestset "Simple Vector Problem" include("Wrappers/fixedpoint_tests__item2.jl")
                    @time @safetestset "Power Method" include("Wrappers/fixedpoint_tests__item3.jl")
                    @time @safetestset "Anderson does not allocate dense Jacobian (#862)" include("Wrappers/fixedpoint_tests__item4.jl")
                    @time @safetestset "LeastSquaresOptim.jl" include("Wrappers/least_squares_tests__item1.jl")
                    @time @safetestset "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Provided" include("Wrappers/least_squares_tests__item2.jl")
                    @time @safetestset "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Not Provided" include("Wrappers/least_squares_tests__item3.jl")
                    @time @safetestset "FastLevenbergMarquardt.jl + StaticArrays" include("Wrappers/least_squares_tests__item4.jl")
                    @time @safetestset "Steady State Problems" include("Wrappers/rootfind_tests__item1.jl")
                    return @time @safetestset "Nonlinear Root Finding Problems" include("Wrappers/rootfind_tests__item2.jl")
                end,
            ),
            "CUDA" => (;
                env = joinpath(@__DIR__, "gpu"),
                body = function ()
                    @time @safetestset "CUDA Tests" include("gpu/cuda_tests__item1.jl")
                    return @time @safetestset "Termination Conditions: Allocations" include("gpu/cuda_tests__item2.jl")
                end,
            ),
        ),
        # QA (Aqua/ExplicitImports via SciMLTesting.run_qa) lives in an isolated sub-env
        # under test/qa so its compat bounds don't constrain the base resolve. Excluded
        # from "All".
        qa = (;
            env = joinpath(@__DIR__, "qa"),
            body = joinpath(@__DIR__, "qa", "qa.jl"),
        ),
        # "All" runs the base-env groups only (Core + PolyAlgorithms + Verbosity); the
        # dep-adding groups and QA run only when selected by name.
        all = ["Core", "PolyAlgorithms", "Verbosity"],
        sublib_env = "NONLINEARSOLVE_TEST_GROUP",
        lib_dir = LIB_DIR,
    )
end
