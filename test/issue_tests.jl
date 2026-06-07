if GROUP == "All" || GROUP == "Core"
    @safetestset "Correct Best Solution: #565" include("issue_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Core"
    @safetestset "Polyalgorithm Fallback Path: CurveFit.jl#76" include("issue_tests__item2.jl")
end
if GROUP == "All" || GROUP == "Core"
    @safetestset "Polyalgorithm Cache solve!: Issue #779" include("issue_tests__item3.jl")
end
