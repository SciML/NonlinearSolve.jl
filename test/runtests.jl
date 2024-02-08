using Pkg, SafeTestsets, XUnit

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env)
    Pkg.activate(env)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@testset runner=ParallelTestRunner() "NonlinearSolve.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "Basic Root Finding Tests" include("core/rootfind.jl")
        @safetestset "Forward AD" include("core/forward_ad.jl")
        @safetestset "Basic NLLS Solvers" include("core/nlls.jl")
        @safetestset "Fixed Point Solvers" include("wrappers/fixedpoint.jl")
        @safetestset "Root Finding Solvers" include("wrappers/rootfind.jl")
        @safetestset "Nonlinear Least Squares Solvers" include("wrappers/nlls.jl")
        @safetestset "23 Test Problems" include("core/23_test_problems.jl")
        @safetestset "Quality Assurance" include("misc/qa.jl")
        @safetestset "Sparsity Tests" include("misc/bruss.jl")
        @safetestset "Polyalgs" include("misc/polyalgs.jl")
        @safetestset "Matrix Resizing" include("misc/matrix_resizing.jl")
        @safetestset "Banded Matrices" include("misc/banded_matrices.jl")
    end

    if GROUP == "GPU"
        activate_env("gpu")
        @safetestset "GPU Tests" include("gpu/core.jl")
    end
end
