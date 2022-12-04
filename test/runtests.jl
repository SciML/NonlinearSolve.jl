using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests + Some AD" begin include("basictests.jl") end
        @time @safetestset "Sparsity Tests" begin include("sparse.jl") end
    end

    if GROUP == "GPU"
        @time @safetestset "GPU Tests" begin include("gpu.jl") end
    end
end
