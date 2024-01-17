using Pkg, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env)
    Pkg.activate(env)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "RootFinding"
        @time @safetestset "Basic Root Finding Tests" include("core/rootfind.jl")
        @time @safetestset "Forward AD" include("core/forward_ad.jl")
    end

    if GROUP == "All" || GROUP == "NLLSSolvers"
        @time @safetestset "Basic NLLS Solvers" include("core/nlls.jl")
    end

    if GROUP == "All" || GROUP == "Wrappers"
        @time @safetestset "Fixed Point Solvers" include("wrappers/fixedpoint.jl")
        @time @safetestset "Root Finding Solvers" include("wrappers/rootfind.jl")
        @time @safetestset "Nonlinear Least Squares Solvers" include("wrappers/nlls.jl")
    end

    if GROUP == "All" || GROUP == "23TestProblems"
        @time @safetestset "23 Test Problems" include("core/23_test_problems.jl")
    end

    if GROUP == "All" || GROUP == "Miscellaneous"
        @time @safetestset "Quality Assurance" include("misc/qa.jl")
        @time @safetestset "Sparsity Tests: Bruss Steady State" include("misc/bruss.jl")
        @time @safetestset "Polyalgs" include("misc/polyalgs.jl")
        @time @safetestset "Matrix Resizing" include("misc/matrix_resizing.jl")
        @time @safetestset "Banded Matrices" include("misc/banded_matrices.jl")
    end

    if GROUP == "GPU"
        activate_env("gpu")
        @time @safetestset "GPU Tests" include("gpu/core.jl")
    end
end
