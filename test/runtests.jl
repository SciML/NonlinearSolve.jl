using ReTestItems, NonlinearSolve, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

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
const ENZYME_ENABLED = VERSION < v"1.12"

const EXTRA_PKGS = Pkg.PackageSpec[]
if GROUP == "all" || GROUP == "downstream"
    push!(EXTRA_PKGS, Pkg.PackageSpec("ModelingToolkit"))
    push!(EXTRA_PKGS, Pkg.PackageSpec("SymbolicIndexingInterface"))
end
if GROUP == "all" || GROUP == "nopre"
    # Only add Enzyme for nopre group if not on prerelease Julia and if enabled
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
