using ReTestItems, NonlinearSolveFirstOrder, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections; strip the prefix
# back to the bare standard section name. Standard sublibrary groups are Core
# (functional/correctness), QA (Aqua/JET/ExplicitImports/allocation) and GPU
# (the dedicated GPU.yml workflow sets GROUP="cuda" directly).
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveFirstOrder"
const GROUP = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

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
    NonlinearSolveFirstOrder;
    tags = _TAGS,
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
