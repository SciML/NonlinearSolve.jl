if GROUP == "all" || GROUP == "wrappers"
    @safetestset "Steady State Problems" include("rootfind_tests__item1.jl")
end
if GROUP == "all" || GROUP == "wrappers"
    @safetestset "Nonlinear Root Finding Problems" include("rootfind_tests__item2.jl")
end
