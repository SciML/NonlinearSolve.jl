if GROUP == "all" || GROUP == "core"
    @safetestset "NLLS Analytic Jacobian" include("core_tests__item1.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "Basic PolyAlgorithms" include("core_tests__item2.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "PolyAlgorithm Type Inference" include("core_tests__item3.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "PolyAlgorithms Autodiff" include("core_tests__item4.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "PolyAlgorithm Aliasing" include("core_tests__item5.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "Ensemble Nonlinear Problems" include("core_tests__item6.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "BigFloat Support" include("core_tests__item7.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "Singular Exception: Issue #153" include("core_tests__item8.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "Simple Scalar Problem: Issue #187" include("core_tests__item9.jl")
end
if GROUP == "all" || GROUP == "downstream"
    @safetestset "Complex Valued Problems: Single-Shooting" include("core_tests__item10.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "No AD" include("core_tests__item11.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "Infeasible" include("core_tests__item12.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "NoInit Caching" include("core_tests__item13.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "Out-of-place Matrix Resizing" include("core_tests__item14.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "Inplace Matrix Resizing" include("core_tests__item15.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "Singular Systems -- Auto Linear Solve Switching" include("core_tests__item16.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "No PolyesterForwardDiff for SArray" include("core_tests__item17.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "NonlinearLeastSquares ReturnCode" include("core_tests__item18.jl")
end
if GROUP == "all" || GROUP == "nopre"
    @safetestset "Default Algorithm Singular Handling" include("core_tests__item19.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "NonNumberEltype error" include("core_tests__item20.jl")
end
if GROUP == "all" || GROUP == "core"
    @safetestset "LinearSolve Preconditioner Interface" include("core_tests__item21.jl")
end
