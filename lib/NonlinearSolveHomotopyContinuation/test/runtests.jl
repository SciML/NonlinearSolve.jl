using NonlinearSolveHomotopyContinuation
using Test
using Aqua

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
