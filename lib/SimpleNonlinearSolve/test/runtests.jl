using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests + Some AD" include("basictests.jl")
        @time @safetestset "Inplace Tests" include("inplace.jl")
        @time @safetestset "Matrix Resizing Tests" include("matrix_resizing_tests.jl")
    end
end
