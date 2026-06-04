using TestItemRunner, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it for parity
# with the other sublibraries. This package has a single untagged Core suite, so
# every section (Core and All) runs the full suite unfiltered, matching the old
# bespoke CI. The "MacOS" suffix only selects a runner, not a different selection.
const _G = get(ENV, "GROUP", "All")
const _SUB = "SciMLJacobianOperators"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const GROUP = lowercase(endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC)

@run_package_tests
