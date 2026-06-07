if GROUP == "all" || GROUP == "cuda"
    @safetestset "CUDA Tests" include("cuda_tests__item1.jl")
end
if GROUP == "all" || GROUP == "cuda"
    @safetestset "Termination Conditions: Allocations" include("cuda_tests__item2.jl")
end
