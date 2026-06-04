using TestItemRunner, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections; strip the prefix
# back to the bare standard section name. Standard sublibrary groups are Core
# (functional/correctness) and QA (Aqua + Explicit Imports).
const _G = get(ENV, "GROUP", "All")
const _SUB = "SciMLJacobianOperators"
const GROUP = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

if GROUP in ("All", "all")
    @run_package_tests
elseif GROUP in ("Core", "core")
    @run_package_tests filter = ti -> (:core in ti.tags)
elseif GROUP in ("QA", "qa")
    @run_package_tests filter = ti -> (:qa in ti.tags)
else
    @run_package_tests filter = ti -> (Symbol(lowercase(GROUP)) in ti.tags)
end
