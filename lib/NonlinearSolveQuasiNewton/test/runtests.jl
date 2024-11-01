using ReTestItems, NonlinearSolveQuasiNewton, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS",
        string(min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4))
    )
)
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1))
    )
)

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    NonlinearSolveQuasiNewton; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
