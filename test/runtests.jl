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

# Centralized sublibrary CI (sublibrary-project-tests.yml@v1) tests each lib/<name>
# via the project model and never routes through this file. This dispatcher only
# matters when the root suite is invoked with a GROUP that names a sublibrary (e.g.
# local `NONLINEARSOLVE_TEST_GROUP=NonlinearSolveFirstOrder julia test/runtests.jl`):
# the bare sublibrary name selects that sublibrary's "Core" group and
# "<sublibrary>_<grp>" selects a named group. We activate the sublibrary's own test
# environment and hand off to its runtests.jl via NONLINEARSOLVE_TEST_GROUP. The
# Pkg.test is done explicitly here (rather than via run_tests's built-in lib_dir
# path) so the Julia < 1.11 transitive [sources] develop walk is preserved verbatim.
base_group, sub_group = detect_sublibrary_group(GROUP, LIB_DIR)

if !isempty(base_group) && isdir(joinpath(LIB_DIR, base_group))
    sublib_path = joinpath(LIB_DIR, base_group)
    Pkg.activate(sublib_path)
    # On Julia < 1.11 the [sources] section is ignored; develop local path deps so
    # CI tests the PR branch code (transitively, including the umbrella root for
    # sublibs that depend back on it). Walk [sources] in case a developed dependency
    # carries its own.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}([normpath(sublib_path)])
        specs = Pkg.PackageSpec[]
        queue = [sublib_path]
        while !isempty(queue)
            pkg_dir = popfirst!(queue)
            toml_path = joinpath(pkg_dir, "Project.toml")
            isfile(toml_path) || continue
            toml = Pkg.TOML.parsefile(toml_path)
            if haskey(toml, "sources")
                for (dep_name, source_spec) in toml["sources"]
                    if source_spec isa Dict && haskey(source_spec, "path")
                        dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                        if isdir(dep_path) && !(dep_path in developed)
                            push!(developed, dep_path)
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        isempty(specs) || Pkg.develop(specs)
    end
    withenv("NONLINEARSOLVE_TEST_GROUP" => sub_group) do
        Pkg.test(
            base_group;
            julia_args = ["--check-bounds=auto"],
            force_latest_compatible_version = false, allow_reresolve = true
        )
    end
elseif GROUP == "Trim"
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
            return @time @safetestset "HomotopyProblem defaults to HomotopySweep when alg is nothing" include("Core/homotopy_sweep_tests__item14.jl")
        end,
        groups = Dict(
            "NoPre" => function ()
                @time @safetestset "Basic PolyAlgorithms" include("NoPre/core_tests__item2.jl")
                @time @safetestset "PolyAlgorithms Autodiff" include("NoPre/core_tests__item4.jl")
                @time @safetestset "Ensemble Nonlinear Problems" include("NoPre/core_tests__item6.jl")
                @time @safetestset "Out-of-place Matrix Resizing" include("NoPre/core_tests__item14.jl")
                @time @safetestset "Inplace Matrix Resizing" include("NoPre/core_tests__item15.jl")
                @time @safetestset "Default Algorithm Singular Handling" include("NoPre/core_tests__item19.jl")
                return @time @safetestset "23 Test Problems: PolyAlgorithms" include("NoPre/23_test_problems_tests__item1.jl")
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
                    return @time @safetestset "maybe_wrap_nonlinear_f skips wrapping inside Enzyme.autodiff (#939)" include("Adjoint/adjoint_tests__item2.jl")
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
        # QA (Aqua/ExplicitImports) lives in an isolated sub-env under test/qa so its
        # compat bounds don't constrain the base resolve. Excluded from "All".
        qa = (;
            env = joinpath(@__DIR__, "qa"),
            body = function ()
                @time @safetestset "Aqua" include("qa/qa.jl")
                return @time @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
            end,
        ),
        # "All" runs the base-env groups only (Core + NoPre + Verbosity); the
        # dep-adding groups and QA run only when selected by name.
        all = ["Core", "NoPre", "Verbosity"],
        sublib_env = "NONLINEARSOLVE_TEST_GROUP",
    )
end
