if GROUP == "All" || GROUP == "Core" || GROUP == "Bounds"
    @safetestset "Bounds: NonlinearLeastSquaresProblem" include("bounds_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Core" || GROUP == "Bounds"
    @safetestset "Bounds: one-sided" include("bounds_tests__item2.jl")
end
if GROUP == "All" || GROUP == "Bounds" || GROUP == "NoPre"
    @safetestset "Bounds: nonlinear model" include("bounds_tests__item3.jl")
end
if GROUP == "All" || GROUP == "Core" || GROUP == "Bounds"
    @safetestset "Bounds: polyalgorithm and quasi-Newton algorithms" include("bounds_tests__item4.jl")
end
