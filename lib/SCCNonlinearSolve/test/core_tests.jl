@safetestset "Manual SCC" include("core_tests__item1.jl")
@safetestset "SCCNonlinearProblem solve without explicit u0 (issue #758)" include("core_tests__item2.jl")
@safetestset "SCC Residuals Transfer" include("core_tests__item3.jl")
@safetestset "Vector-form SCCNonlinearProblem with FunctionWrappers" include("core_tests__item4.jl")
