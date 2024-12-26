using NonlinearSolveHomotopyContinuation
using Test
using Aqua

@testset "NonlinearSolveHomotopyContinuation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NonlinearSolveHomotopyContinuation)
    end
    # Write your tests here.
end
