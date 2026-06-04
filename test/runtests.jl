using ReTestItems, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Detect sublibrary test groups.
# The centralized SublibraryCI (sublibrary-tests.yml@v1) runs `Pkg.test` on this
# root package with GROUP set to a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any other group (e.g. QA, GPU). We read the RAW
# GROUP env here, before the main-package logic lowercases/parses it, so that a
# sublibrary dispatch always uses the unmodified value.
const RAW_GROUP = get(ENV, "GROUP", "All")
const _LIB_DIR = joinpath(dirname(@__DIR__), "lib")

# Check if GROUP matches a sublibrary, possibly with a _SUFFIX for the test group.
# Scan underscores right-to-left to find the longest matching sublibrary prefix.
function _detect_sublibrary_group(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end
const _BASE_GROUP, _TEST_GROUP = _detect_sublibrary_group(RAW_GROUP, _LIB_DIR)

if isdir(joinpath(_LIB_DIR, _BASE_GROUP))
    Pkg.activate(joinpath(_LIB_DIR, _BASE_GROUP))
    # On Julia < 1.11, the [sources] section in Project.toml is not supported.
    # Manually Pkg.develop local path dependencies so CI tests the PR branch code.
    # We resolve transitively: each developed dependency's own [sources] are also
    # developed, so that packages like a low-level sublibrary (a source dependency
    # of a higher-level one) are correctly found even when testing the higher one.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}()
        # Never develop the active project: when sublibraries cyclically
        # reference each other via [sources], the transitive walk below would
        # otherwise try to `Pkg.develop` the active project itself, which
        # Pkg refuses with "package <X> has the same name or UUID as the
        # active project".
        push!(developed, normpath(joinpath(_LIB_DIR, _BASE_GROUP)))
        specs = Pkg.PackageSpec[]
        queue = [joinpath(_LIB_DIR, _BASE_GROUP)]
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
                            @info "Queuing local source dependency" dep_name dep_path
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            # Queue this dependency so its own [sources] are also resolved.
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        # Batch the develop call so Pkg resolves all path deps together;
        # calling it one-at-a-time would re-resolve the active project and
        # fail to find unregistered siblings.
        isempty(specs) || Pkg.develop(specs)
    end
    withenv("NLS_TEST_GROUP" => _TEST_GROUP) do
        Pkg.test(_BASE_GROUP, julia_args = ["--check-bounds=auto", "--depwarn=yes"], force_latest_compatible_version = false, allow_reresolve = true)
    end
else
    function parse_test_args()
        test_args_from_env = @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS
        test_args = Dict{String, String}()
        for arg in test_args_from_env
            if contains(arg, "=")
                key, value = split(arg, "="; limit = 2)
                test_args[key] = value
            end
        end
        @info "Parsed test args" test_args
        return test_args
    end

    const PARSED_TEST_ARGS = parse_test_args()

    function get_from_test_args_or_env(key, default)
        haskey(PARSED_TEST_ARGS, key) && return PARSED_TEST_ARGS[key]
        return get(ENV, key, default)
    end

    const GROUP = lowercase(get_from_test_args_or_env("GROUP", "all"))

    # Disable Enzyme on Julia 1.12+ due to compatibility issues
    # To re-enable: change condition to `true` or `VERSION < v"1.13"`
    const ENZYME_ENABLED = VERSION < v"1.13"

    if GROUP != "trim"
        using NonlinearSolve  # trimming uses NonlinearSolve from a custom environment

        const EXTRA_PKGS = Pkg.PackageSpec[]
        if GROUP == "all" || GROUP == "downstream"
            push!(EXTRA_PKGS, Pkg.PackageSpec("ModelingToolkit"))
            push!(EXTRA_PKGS, Pkg.PackageSpec("SymbolicIndexingInterface"))
            push!(EXTRA_PKGS, Pkg.PackageSpec("OrdinaryDiffEqTsit5"))
        end
        if GROUP in ("all", "nopre", "bounds")
            # Only add Enzyme for specific groups if not on prerelease Julia and if enabled
            if isempty(VERSION.prerelease) && ENZYME_ENABLED
                push!(EXTRA_PKGS, Pkg.PackageSpec("Enzyme"))
                push!(EXTRA_PKGS, Pkg.PackageSpec("Mooncake"))
                push!(EXTRA_PKGS, Pkg.PackageSpec("SciMLSensitivity"))
            end
        end
        if GROUP == "all" || GROUP == "cuda"
            # Only add CUDA for cuda group if not on prerelease Julia
            if isempty(VERSION.prerelease)
                push!(EXTRA_PKGS, Pkg.PackageSpec("CUDA"))
            end
        end
        length(EXTRA_PKGS) ≥ 1 && Pkg.add(EXTRA_PKGS)

        # Use sequential execution for wrapper tests to avoid parallel initialization issues
        const RETESTITEMS_NWORKERS = if GROUP == "wrappers"
            0  # Sequential execution for wrapper tests
        else
            tmp = get(ENV, "RETESTITEMS_NWORKERS", "")
            isempty(tmp) &&
                (tmp = string(min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4)))
            parse(Int, tmp)
        end
        const RETESTITEMS_NWORKER_THREADS = begin
            tmp = get(ENV, "RETESTITEMS_NWORKER_THREADS", "")
            isempty(tmp) &&
                (tmp = string(max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1)))
            parse(Int, tmp)
        end

        @info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

        ReTestItems.runtests(
            NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
            nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
            testitem_timeout = 3600
        )
    elseif GROUP == "trim" && VERSION >= v"1.12.0-rc1"  # trimming has been introduced in julia 1.12
        Pkg.activate(joinpath(dirname(@__FILE__), "trim"))
        Pkg.instantiate()
        include("trim/runtests.jl")
    end
end
