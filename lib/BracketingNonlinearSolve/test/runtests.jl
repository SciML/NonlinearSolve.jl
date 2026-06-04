using TestItemRunner, InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it back to a
# bare section name. The old bespoke CI ran the full suite (every tag) on each leg,
# so Core/All run everything; an explicit group filters by tag. The "MacOS" suffix
# only selects a runner, not a different selection.
const _G = get(ENV, "GROUP", "All")
const _SUB = "BracketingNonlinearSolve"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const _SEC_BASE = endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC
const GROUP = lowercase(_SEC_BASE)

@testset "BracketingNonlinearSolve.jl" begin
    if GROUP == "all" || GROUP == "core"
        @run_package_tests
    else
        @run_package_tests filter = ti -> (Symbol(GROUP) in ti.tags)
    end
end
