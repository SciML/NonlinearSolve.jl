if GROUP == "All" || GROUP == "NoPre"
    @safetestset "ForwardDiff.jl Integration" include("forward_ad_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Core"
    @safetestset "NLLS Hessian SciML/NonlinearSolve.jl#445" include("forward_ad_tests__item2.jl")
end
if GROUP == "All" || GROUP == "Core"
    @safetestset "reinit! on ForwardDiff cache SciML/NonlinearSolve.jl#391" include("forward_ad_tests__item3.jl")
end
