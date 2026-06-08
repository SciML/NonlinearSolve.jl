using Pkg
using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

@info "Running tests for group: $(GROUP)"

# QA tooling (Aqua/ExplicitImports) lives in an isolated sub-environment under
# test/qa so its compat bounds don't constrain the main test resolve. Develop
# the in-repo path deps so [sources] also works on Julia < 1.11 (where the
# Project.toml [sources] table is ignored), then instantiate.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop([
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "NonlinearSolveBase")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "SciMLJacobianOperators")),
        ])
    end
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    include("inference_tests.jl")
    include("least_squares_tests.jl")
    include("misc_tests.jl")
    include("rootfind_tests.jl")
    include("sparsity_tests.jl")
    # QA runs last: activate_qa_env() switches the active project to test/qa.
    activate_qa_env()
    @safetestset "Aqua" include("qa/qa.jl")
    @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
end
