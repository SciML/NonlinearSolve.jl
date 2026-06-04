using ReTestItems, NonlinearSolveQuasiNewton, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it back to a
# bare section name so the dispatch below (and bare "All" for local runs) keeps
# working; the "MacOS" suffix only selects a runner, not a different selection.
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveQuasiNewton"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const _SEC_BASE = endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC
const GROUP = lowercase(_SEC_BASE)

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
    NonlinearSolveQuasiNewton;
    tags = (GROUP == "all" || GROUP == "core" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
