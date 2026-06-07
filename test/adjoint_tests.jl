if GROUP == "all" || GROUP == "nopre"
    @safetestset "Adjoint Tests" include("adjoint_tests__item1.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "maybe_wrap_nonlinear_f skips wrapping inside Enzyme.autodiff (#939)" include("adjoint_tests__item2.jl")
end
