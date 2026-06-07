if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Steady State Problems" include("rootfind_tests__item1.jl")
end
if GROUP == "All" || GROUP == "Wrappers"
    @safetestset "Nonlinear Root Finding Problems" include("rootfind_tests__item2.jl")
end
