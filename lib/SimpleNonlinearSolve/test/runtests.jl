using ReTestItems, SimpleNonlinearSolve, Hwloc, InteractiveUtils
using Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

if GROUP == "adjoint" || GROUP == "all"
    Pkg.add(["SciMLSensitivity"])
end

if GROUP == "cuda" || GROUP == "all"
    Pkg.add(["CUDA"])
end

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    SimpleNonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600, retries = 4)
