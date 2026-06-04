using TestItemRunner, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# The root NonlinearSolve runtests dispatcher activates this sublibrary and sets
# NLS_TEST_GROUP to the bare standard section name. Standard sublibrary groups
# are Core (functional/correctness) and QA (Aqua + Explicit Imports).
const GROUP = get(ENV, "NLS_TEST_GROUP", "All")

if GROUP in ("All", "all")
    @run_package_tests
elseif GROUP in ("Core", "core")
    @run_package_tests filter = ti -> (:core in ti.tags)
elseif GROUP in ("QA", "qa")
    @run_package_tests filter = ti -> (:qa in ti.tags)
else
    @run_package_tests filter = ti -> (Symbol(lowercase(GROUP)) in ti.tags)
end
