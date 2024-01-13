using Pkg, SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env)
    Pkg.activate(env)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time @testset "SimpleNonlinearSolve.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("basictests.jl")
        @time @safetestset "Forward AD" include("forward_ad.jl")
        @time @safetestset "Matrix Resizing Tests" include("matrix_resizing_tests.jl")
        @time @safetestset "Least Squares Tests" include("least_squares.jl")
        @time @safetestset "23 Test Problems" include("23_test_problems.jl")
    end

    if GROUP == "CUDA"
        activate_env("cuda")
        @time @safetestset "CUDA Tests" include("cuda.jl")
    end
end
