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
    parse(
        Int, get(ENV, "RETESTITEMS_NWORKERS",
            string(min(ifelse(Sys.iswindows(), 0, max(Hwloc.num_physical_cores(), 1)), 4))
        )
    )
end
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(max(Hwloc.num_virtual_cores(), 1) ÷ max(RETESTITEMS_NWORKERS, 1), 1))
    )
)

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
