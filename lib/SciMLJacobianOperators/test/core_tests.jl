@safetestset "Scalar Ops" include("core_tests__item1.jl")
@safetestset "Inplace Problems" include("core_tests__item2.jl")
@safetestset "Out-of-place Problems" include("core_tests__item3.jl")
@safetestset "Copy with Tuple parameters (Issue #752)" include("core_tests__item6.jl")
