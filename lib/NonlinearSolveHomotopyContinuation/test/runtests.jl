using NonlinearSolveHomotopyContinuation
using Test
using Aqua

# The root NonlinearSolve runtests dispatcher activates this sublibrary and sets
# NLS_TEST_GROUP to the bare standard section name. Standard sublibrary groups
# are Core (functional/correctness — AllRoots/Single Root) and QA (Aqua).
const GROUP = get(ENV, "NLS_TEST_GROUP", "All")

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
