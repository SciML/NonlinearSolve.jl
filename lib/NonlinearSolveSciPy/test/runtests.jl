using ReTestItems, NonlinearSolveSciPy, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# The root NonlinearSolve runtests dispatcher activates this sublibrary and sets
# NLS_TEST_GROUP to the bare standard section name. Standard sublibrary groups
# are Core (functional/correctness — the SciPy wrappers + basic load tests) and
# QA. This sublibrary has no QA-tagged items, so the QA leg runs nothing.
const GROUP = get(ENV, "NLS_TEST_GROUP", "All")

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
