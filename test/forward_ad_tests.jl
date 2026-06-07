if GROUP == "all" || GROUP == "nopre"
    @safetestset "ForwardDiff.jl Integration" include("forward_ad_tests__item1.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "NLLS Hessian SciML/NonlinearSolve.jl#445" include("forward_ad_tests__item2.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "reinit! on ForwardDiff cache SciML/NonlinearSolve.jl#391" include("forward_ad_tests__item3.jl")
end
