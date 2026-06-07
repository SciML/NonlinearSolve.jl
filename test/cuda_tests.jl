if GROUP == "All" || GROUP == "CUDA"
    @safetestset "CUDA Tests" include("cuda_tests__item1.jl")
end
if GROUP == "All" || GROUP == "CUDA"
    @safetestset "Termination Conditions: Allocations" include("cuda_tests__item2.jl")
end
