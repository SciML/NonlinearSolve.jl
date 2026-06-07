if GROUP == "all" || GROUP == "core" || GROUP == "bounds"
    @safetestset "Bounds: NonlinearLeastSquaresProblem" include("bounds_tests__item1.jl")
end
if GROUP == "all" || GROUP == "core" || GROUP == "bounds"
    @safetestset "Bounds: one-sided" include("bounds_tests__item2.jl")
end
if GROUP == "all" || GROUP == "bounds" || GROUP == "nopre"
    @safetestset "Bounds: nonlinear model" include("bounds_tests__item3.jl")
end
if GROUP == "all" || GROUP == "core" || GROUP == "bounds"
    @safetestset "Bounds: polyalgorithm and quasi-Newton algorithms" include("bounds_tests__item4.jl")
end
