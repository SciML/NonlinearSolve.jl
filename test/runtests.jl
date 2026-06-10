using Pkg
using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch. SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; the root CI sets
# GROUP. Read either so the same runtests.jl works as both the root test entry
# and the sublibrary dispatcher.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

# Match sublibrary dirs case-insensitively.
function _find_lib(base, lib_dir)
    isdir(lib_dir) || return nothing
    for d in readdir(lib_dir)
        isdir(joinpath(lib_dir, d)) && lowercase(d) == lowercase(base) && return d
    end
    return nothing
end

# If GROUP names a lib/<X> sublibrary (optionally `<X>_<TESTGROUP>`), activate
# that sublibrary, develop its in-repo [sources], and Pkg.test it with the
# sublibrary's own group env var set. Mirrors OrdinaryDiffEq's root dispatcher
# so a single GROUP value can target any sublibrary. Scan underscores
# right-to-left for the longest matching sublibrary prefix.
function _detect_sublibrary_group(group, lib_dir)
    _find_lib(group, lib_dir) !== nothing && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && _find_lib(group[1:(i - 1)], lib_dir) !== nothing
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end

# In-repo path dependencies developed when a sub-environment's [sources] table
# is ignored (Julia < 1.11). Lists the umbrella root and every sublibrary so any
# group env that references a subset still resolves to the PR branch code.
function _develop_inrepo_sources()
    VERSION < v"1.11.0-DEV.0" || return
    root = dirname(@__DIR__)
    lib = joinpath(root, "lib")
    specs = Pkg.PackageSpec[Pkg.PackageSpec(path = root)]
    for d in readdir(lib)
        isdir(joinpath(lib, d)) && push!(specs, Pkg.PackageSpec(path = joinpath(lib, d)))
    end
    return Pkg.develop(specs)
end

# Activate a dep-adding group's isolated sub-environment under test/<Group> and
# instantiate it. These groups carry deps beyond the base test set (and are
# excluded from the All run, which uses the base env). On Julia < 1.11 the
# [sources] table is ignored, so the in-repo path deps are developed first.
function activate_group_env(group)
    Pkg.activate(joinpath(@__DIR__, group))
    _develop_inrepo_sources()
    return Pkg.instantiate()
end

@time begin
    lib_dir = joinpath(dirname(@__DIR__), "lib")
    base_group, sub_group = _detect_sublibrary_group(GROUP, lib_dir)
    sublib = _find_lib(base_group, lib_dir)

    if sublib !== nothing
        sub_path = joinpath(lib_dir, sublib)
        Pkg.activate(sub_path)
        # On Julia < 1.11 the [sources] section is ignored; develop local path
        # deps so CI tests the PR branch code (transitively, including the
        # umbrella root for sublibs that depend back on it).
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}([normpath(sub_path)])
            specs = Pkg.PackageSpec[]
            queue = [sub_path]
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
                sublib;
                julia_args = ["--check-bounds=auto"],
                force_latest_compatible_version = false, allow_reresolve = true
            )
        end
    elseif GROUP == "Trim" && VERSION >= v"1.12.0-rc1"
        # Trimming was introduced in Julia 1.12; runs in its own environment.
        Pkg.activate(joinpath(@__DIR__, "trim"))
        Pkg.instantiate()
        include("trim/runtests.jl")
    else
        @info "Running tests for group: $(GROUP)"

        # --- Base-env groups (no extra deps; part of the All run) ---
        if GROUP == "All" || GROUP == "Core"
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
            @time @safetestset "HomotopyProblem defaults to HomotopySweep when alg is nothing" include("Core/homotopy_sweep_tests__item14.jl")
        end

        if GROUP == "All" || GROUP == "NoPre"
            @time @safetestset "Basic PolyAlgorithms" include("NoPre/core_tests__item2.jl")
            @time @safetestset "PolyAlgorithms Autodiff" include("NoPre/core_tests__item4.jl")
            @time @safetestset "Ensemble Nonlinear Problems" include("NoPre/core_tests__item6.jl")
            @time @safetestset "Out-of-place Matrix Resizing" include("NoPre/core_tests__item14.jl")
            @time @safetestset "Inplace Matrix Resizing" include("NoPre/core_tests__item15.jl")
            @time @safetestset "Default Algorithm Singular Handling" include("NoPre/core_tests__item19.jl")
            @time @safetestset "23 Test Problems: PolyAlgorithms" include("NoPre/23_test_problems_tests__item1.jl")
        end

        if GROUP == "All" || GROUP == "Verbosity"
            @time @safetestset "Nonlinear Verbosity" include("Verbosity/verbosity_tests__item1.jl")
        end

        # --- Dep-adding groups (isolated sub-envs; excluded from All) ---
        if GROUP == "Downstream"
            activate_group_env("Downstream")
            @time @safetestset "Complex Valued Problems: Single-Shooting" include("Downstream/core_tests__item10.jl")
            @time @safetestset "Modeling Toolkit Cache Indexing" include("Downstream/mtk_cache_indexing_tests__item1.jl")
        end

        if GROUP == "Bounds"
            activate_group_env("Bounds")
            @time @safetestset "Bounds: nonlinear model" include("Bounds/bounds_tests__item3.jl")
        end

        if GROUP == "Adjoint"
            activate_group_env("Adjoint")
            @time @safetestset "Adjoint Tests" include("Adjoint/adjoint_tests__item1.jl")
            @time @safetestset "maybe_wrap_nonlinear_f skips wrapping inside Enzyme.autodiff (#939)" include("Adjoint/adjoint_tests__item2.jl")
        end

        if GROUP == "Wrappers"
            activate_group_env("Wrappers")
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
            @time @safetestset "Nonlinear Root Finding Problems" include("Wrappers/rootfind_tests__item2.jl")
        end

        if GROUP == "CUDA"
            activate_group_env("gpu")
            @time @safetestset "CUDA Tests" include("gpu/cuda_tests__item1.jl")
            @time @safetestset "Termination Conditions: Allocations" include("gpu/cuda_tests__item2.jl")
        end

        # QA (Aqua/ExplicitImports) lives in an isolated sub-env under test/qa so
        # its compat bounds don't constrain the base resolve. Excluded from All.
        if GROUP == "QA"
            activate_group_env("qa")
            @time @safetestset "Aqua" include("qa/qa.jl")
            @time @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
        end
    end
end # @time
