using ReTestItems, NonlinearSolve, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@info "Running tests for group: $LUXLIB_TEST_GROUP with $RETESTITEMS_NWORKERS workers"

ReTestItems.runtests(NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS,
    nworker_threads = RETESTITEMS_NWORKER_THREADS, testitem_timeout = 3600, retries=4)
