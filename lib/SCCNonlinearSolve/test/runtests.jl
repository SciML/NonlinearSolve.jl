using ReTestItems, SCCNonlinearSolve, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# The root NonlinearSolve runtests dispatcher activates this sublibrary and sets
# NLS_TEST_GROUP to the bare standard section name. Standard sublibrary groups
# are Core (functional/correctness), QA (Aqua/JET/ExplicitImports/allocation) and
# GPU (the dedicated GPU.yml workflow sets GROUP="cuda" directly).
const GROUP = get(ENV, "NLS_TEST_GROUP", "All")

const _TAGS = GROUP in ("All", "all") ? nothing :
    GROUP in ("Core", "core") ? [:core] :
    GROUP in ("QA", "qa") ? [:qa] :
    GROUP in ("GPU", "gpu", "cuda") ? [:cuda] : [Symbol(lowercase(GROUP))]

const RETESTITEMS_NWORKERS = parse(
    Int, get(
        ENV, "RETESTITEMS_NWORKERS",
        string(min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4))
    )
)
const RETESTITEMS_NWORKER_THREADS = parse(
    Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1))
    )
)

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    SCCNonlinearSolve;
    tags = _TAGS,
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
