if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "LeastSquaresOptim.jl" include("least_squares_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Provided" include("least_squares_tests__item2.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Not Provided" include("least_squares_tests__item3.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "FastLevenbergMarquardt.jl + StaticArrays" include("least_squares_tests__item4.jl")
end
