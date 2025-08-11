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
    default_workers = try
        min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4)
    catch
        # Fallback if Hwloc fails (e.g., on some CI environments)
        0
    end
    parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(default_workers)))
end
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(try
            max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1)
        catch
            # Fallback if Hwloc fails
            1
        end)
    )
)

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
