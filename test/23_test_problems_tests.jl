if GROUP == "all" || GROUP == "nopre"
    @safetestset "23 Test Problems: PolyAlgorithms" include("23_test_problems_tests__item1.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: NewtonRaphson" include("23_test_problems_tests__item2.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: Halley" include("23_test_problems_tests__item3.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: TrustRegion" include("23_test_problems_tests__item4.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: LevenbergMarquardt" include("23_test_problems_tests__item5.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: DFSane" include("23_test_problems_tests__item6.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: Broyden" include("23_test_problems_tests__item7.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: Klement" include("23_test_problems_tests__item8.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "23 Test Problems: PseudoTransient" include("23_test_problems_tests__item9.jl")
end
