using ReTestItems, NonlinearSolve, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

const EXTRA_PKGS = Pkg.PackageSpec[]
if GROUP == "all" || GROUP == "downstream"
    push!(EXTRA_PKGS, Pkg.PackageSpec("ModelingToolkit"))
    push!(EXTRA_PKGS, Pkg.PackageSpec("SymbolicIndexingInterface"))
end
if GROUP == "all" || GROUP == "nopre"
    # Only add Enzyme for nopre group if not on prerelease Julia
    if isempty(VERSION.prerelease)
        push!(EXTRA_PKGS, Pkg.PackageSpec("Enzyme"))
    end
end
length(EXTRA_PKGS) ≥ 1 && Pkg.add(EXTRA_PKGS)

# Use sequential execution for wrapper tests to avoid parallel initialization issues
const RETESTITEMS_NWORKERS = if GROUP == "wrappers"
    0  # Sequential execution for wrapper tests
else
    # Ensure we have a valid default value even if Hwloc fails
    default_workers = try
        min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4)
    catch
        1  # Fallback to 1 worker if Hwloc fails
    end
    parse(
        Int, get(ENV, "RETESTITEMS_NWORKERS", string(default_workers))
    )
end
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(
            try
                Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1)
            catch
                1  # Fallback to 1 thread if Hwloc fails
            end,
            1
        ))
    )
)

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
