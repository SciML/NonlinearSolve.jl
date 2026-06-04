using NonlinearSolveHomotopyContinuation
using Test
using Aqua

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it for parity
# with the other sublibraries. This suite is a single always-on Core block (no group
# filtering), so every section runs it, matching the old bespoke CI. The "MacOS"
# suffix only selects a runner, not a different selection.
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveHomotopyContinuation"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const GROUP = lowercase(endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC)

@testset "NonlinearSolveHomotopyContinuation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NonlinearSolveHomotopyContinuation)
    end
    @testset "AllRoots" begin
        include("allroots.jl")
    end
    @testset "Single Root" begin
        include("single_root.jl")
    end
end
