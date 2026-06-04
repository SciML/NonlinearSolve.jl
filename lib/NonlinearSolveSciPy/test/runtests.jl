using ReTestItems, NonlinearSolveSciPy, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections; strip the prefix
# back to the bare standard section name. Standard sublibrary groups are Core
# (functional/correctness — the SciPy wrappers + basic load tests) and QA. This
# sublibrary has no QA-tagged items, so the QA leg runs nothing.
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveSciPy"
const GROUP = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

const _TAGS = GROUP in ("All", "all") ? nothing :
    GROUP in ("Core", "core") ? [:core] :
    GROUP in ("QA", "qa") ? [:qa] :
    GROUP in ("GPU", "gpu", "cuda") ? [:cuda] : [Symbol(lowercase(GROUP))]

const NWORKERS = 1   # PythonCall is not thread-safe across multiple Julia processes/threads
const NTHREADS = 1

@info "Running SciPy solver tests serially (nworkers = 1) to avoid Python multithreading issues"

ReTestItems.runtests(
    NonlinearSolveSciPy;
    tags = _TAGS,
    nworkers = NWORKERS, nworker_threads = NTHREADS,
    testitem_timeout = 3600
)
