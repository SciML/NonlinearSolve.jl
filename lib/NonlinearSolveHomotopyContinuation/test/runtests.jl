using NonlinearSolveHomotopyContinuation
using Test
using Aqua

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections; strip the prefix
# back to the bare standard section name. Standard sublibrary groups are Core
# (functional/correctness — AllRoots/Single Root) and QA (Aqua).
const _G = get(ENV, "GROUP", "All")
const _SUB = "NonlinearSolveHomotopyContinuation"
const GROUP = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

const _RUN_CORE = GROUP in ("All", "all", "Core", "core")
const _RUN_QA = GROUP in ("All", "all", "QA", "qa")

@testset "NonlinearSolveHomotopyContinuation.jl" begin
    if _RUN_QA
        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(NonlinearSolveHomotopyContinuation)
        end
    end
    if _RUN_CORE
        @testset "AllRoots" begin
            include("allroots.jl")
        end
        @testset "Single Root" begin
            include("single_root.jl")
        end
    end
end
