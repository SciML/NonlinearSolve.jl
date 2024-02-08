using Pkg, SafeTestsets, XUnit

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env)
    Pkg.activate(env)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@testset runner=ParallelTestRunner() "SimpleNonlinearSolve.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "Basic Tests" include("basictests.jl")
        @safetestset "Forward AD" include("forward_ad.jl")
        @safetestset "Matrix Resizing Tests" include("matrix_resizing_tests.jl")
        @safetestset "Least Squares Tests" include("least_squares.jl")
        @safetestset "23 Test Problems" include("23_test_problems.jl")
        @safetestset "Simple Adjoint Tests" include("adjoint.jl")
    end

    if GROUP == "CUDA"
        activate_env("cuda")
        @safetestset "CUDA Tests" include("cuda.jl")
    end
end
