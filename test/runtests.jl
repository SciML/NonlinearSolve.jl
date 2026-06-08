using Pkg
using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch. SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; the root CI sets
# GROUP. Read either so the same runtests.jl works as both the root test entry
# and the sublibrary dispatcher.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

# Disable Enzyme on Julia 1.13+ due to compatibility issues.
const ENZYME_ENABLED = VERSION < v"1.13"

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

# QA tooling (Aqua/ExplicitImports) lives in an isolated sub-environment under
# test/qa so its compat bounds don't constrain the main test resolve. Develop
# the in-repo path deps (the umbrella package and its sublibraries) so [sources]
# also works on Julia < 1.11 (where the Project.toml [sources] table is
# ignored), then instantiate.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    if VERSION < v"1.11.0-DEV.0"
        root = dirname(@__DIR__)
        lib = joinpath(root, "lib")
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(lib, "BracketingNonlinearSolve")),
            Pkg.PackageSpec(path = joinpath(lib, "NonlinearSolveBase")),
            Pkg.PackageSpec(path = joinpath(lib, "NonlinearSolveFirstOrder")),
            Pkg.PackageSpec(path = joinpath(lib, "NonlinearSolveQuasiNewton")),
            Pkg.PackageSpec(path = joinpath(lib, "NonlinearSolveSpectralMethods")),
            Pkg.PackageSpec(path = joinpath(lib, "SimpleNonlinearSolve")),
        ])
    end
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
        # Heavy/optional group deps are added on demand so the default Core
        # matrix stays lightweight (matches the original ReTestItems setup).
        extra_pkgs = Pkg.PackageSpec[]
        if GROUP == "All" || GROUP == "Downstream"
            push!(extra_pkgs, Pkg.PackageSpec("ModelingToolkit"))
            push!(extra_pkgs, Pkg.PackageSpec("SymbolicIndexingInterface"))
            push!(extra_pkgs, Pkg.PackageSpec("OrdinaryDiffEqTsit5"))
        end
        if GROUP in ("All", "NoPre", "Bounds")
            if isempty(VERSION.prerelease) && ENZYME_ENABLED
                push!(extra_pkgs, Pkg.PackageSpec("Enzyme"))
                push!(extra_pkgs, Pkg.PackageSpec("Mooncake"))
                push!(extra_pkgs, Pkg.PackageSpec("SciMLSensitivity"))
            end
        end
        if GROUP == "All" || GROUP == "CUDA"
            isempty(VERSION.prerelease) && push!(extra_pkgs, Pkg.PackageSpec("CUDA"))
        end
        isempty(extra_pkgs) || Pkg.add(extra_pkgs)

        @info "Running tests for group: $(GROUP)"

        # Each test file is a sequence of top-level, per-item group-guarded
        # @safetestset includes, so we include them all and the active GROUP
        # selects which items run.
        @time include("core_tests.jl")
        @time include("23_test_problems_tests.jl")
        @time include("bounds_tests.jl")
        @time include("default_alg_tests.jl")
        @time include("forward_ad_tests.jl")
        @time include("issue_tests.jl")
        @time include("adjoint_tests.jl")
        @time include("mtk_cache_indexing_tests.jl")
        @time include("verbosity_tests.jl")
        @time include("cuda_tests.jl")
        @time include("wrappers/fixedpoint_tests.jl")
        @time include("wrappers/least_squares_tests.jl")
        @time include("wrappers/rootfind_tests.jl")

        # QA runs last: activate_qa_env() switches the active project to test/qa.
        # Gated to the Misc group (and All) to preserve the prior dispatch.
        if GROUP == "All" || GROUP == "Misc"
            activate_qa_env()
            @time @safetestset "Aqua" include("qa/qa.jl")
            @time @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
        end
    end
end # @time
