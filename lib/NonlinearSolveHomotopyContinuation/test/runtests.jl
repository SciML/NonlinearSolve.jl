using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = lowercase(get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "all")))

@info "Running tests for group: $(GROUP)"

if GROUP == "all" || GROUP == "core"
    @safetestset "Code quality (Aqua.jl)" begin
        using NonlinearSolveHomotopyContinuation, Aqua
        Aqua.test_all(NonlinearSolveHomotopyContinuation)
    end
    @safetestset "AllRoots" include("allroots.jl")
    @safetestset "Single Root" include("single_root.jl")
end
