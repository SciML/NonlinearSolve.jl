using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function activate_downstream_env()
    Pkg.activate("GPU")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests + Some AD" include("basictests.jl")
    end

    if GROUP == "All" || GROUP == "NLLS"
        @time @safetestset "Nonlinear Least Squares" include("nonlinear_least_squares.jl")
    end

    if GROUP == "All" || GROUP == "Wrappers"
        @time @safetestset "MINPACK" include("minpack.jl")
        @time @safetestset "NLsolve" include("nlsolve.jl")
        @time @safetestset "SIAMFANLEquations" include("siamfanlequations.jl")
    end

    if GROUP == "All" || GROUP == "23TestProblems"
        @time @safetestset "23 Test Problems" include("23_test_problems.jl")
    end

    if GROUP == "All" || GROUP == "Miscellaneous"
        @time @safetestset "Quality Assurance" include("qa.jl")
        @time @safetestset "Sparsity Tests" include("sparse.jl")
        @time @safetestset "Polyalgs" include("polyalgs.jl")
        @time @safetestset "Matrix Resizing" include("matrix_resizing.jl")
        @time @safetestset "Infeasible Problems" include("infeasible.jl")
    end

    if GROUP == "GPU"
        activate_downstream_env()
        @time @safetestset "GPU Tests" include("gpu.jl")
    end
end
