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
        ])
    end
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    include("muller_tests.jl")
end

if GROUP == "All" || GROUP == "Adjoint"
    include("adjoint_tests.jl")
end

if GROUP == "All" || GROUP == "Core"
    include("rootfind_tests.jl")
end

# QA (Aqua/ExplicitImports) is a dep-adding group: it runs in its own isolated
# sub-env under test/qa (excluded from the base/Core/All run).
if GROUP == "QA"
    activate_qa_env()
    @safetestset "Aqua" include("qa/qa.jl")
    @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
end
