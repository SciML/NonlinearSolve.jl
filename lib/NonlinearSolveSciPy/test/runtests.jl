using ReTestItems, NonlinearSolveSciPy, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

const NWORKERS = 1   # PythonCall is not thread-safe across multiple Julia processes/threads
const NTHREADS = 1

@info "Running SciPy solver tests serially (nworkers = 1) to avoid Python multithreading issues"

ReTestItems.runtests(
    NonlinearSolveSciPy; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = NWORKERS, nworker_threads = NTHREADS,
    testitem_timeout = 3600
)
