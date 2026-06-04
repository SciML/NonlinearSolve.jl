using ReTestItems, NonlinearSolveSciPy, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it back to a
# bare section name so the dispatch below (and bare "All" for local runs) keeps
# working; the "MacOS" suffix only selects a runner, not a different selection.
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveSciPy"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const _SEC_BASE = endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC
const GROUP = lowercase(_SEC_BASE)

const NWORKERS = 1   # PythonCall is not thread-safe across multiple Julia processes/threads
const NTHREADS = 1

@info "Running SciPy solver tests serially (nworkers = 1) to avoid Python multithreading issues"

ReTestItems.runtests(
    NonlinearSolveSciPy;
    tags = (GROUP == "all" || GROUP == "core" ? nothing : [Symbol(GROUP)]),
    nworkers = NWORKERS, nworker_threads = NTHREADS,
    testitem_timeout = 3600
)
