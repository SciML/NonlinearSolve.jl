if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Simple Scalar Problem" include("fixedpoint_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Simple Vector Problem" include("fixedpoint_tests__item2.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Power Method" include("fixedpoint_tests__item3.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Anderson does not allocate dense Jacobian (#862)" include("fixedpoint_tests__item4.jl")
end
