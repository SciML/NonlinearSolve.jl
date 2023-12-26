using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@time @testset "SimpleNonlinearSolve.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("basictests.jl")
        @time @safetestset "Forward AD" include("forward_ad.jl")
        @time @safetestset "Matrix Resizing Tests" include("matrix_resizing_tests.jl")
        @time @safetestset "Least Squares Tests" include("least_squares.jl")
        @time @safetestset "23 Test Problems" include("23_test_problems.jl")
    end
end
