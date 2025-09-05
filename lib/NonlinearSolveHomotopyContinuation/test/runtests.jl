using NonlinearSolveHomotopyContinuation
using Test
using Aqua

@testset "NonlinearSolveHomotopyContinuation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NonlinearSolveHomotopyContinuation; persistent_tasks = false)
        Aqua.test_persistent_tasks(
            NonlinearSolveHomotopyContinuation; broken = VERSION < v"1.11")
    end
    @testset "AllRoots" begin
        include("allroots.jl")
    end
    @testset "Single Root" begin
        include("single_root.jl")
    end
end
