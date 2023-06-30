using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests + Some AD" begin
            include("basictests.jl")
        end

        @time @safetestset "Inplace Tests" begin
            include("inplace.jl")
        end
    end
end
