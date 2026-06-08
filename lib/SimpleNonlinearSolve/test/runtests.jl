using SafeTestsets, Test, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

@info "Running tests for group: $(GROUP)"

# Activate a dep-adding group's isolated sub-environment under test/<dir> and
# instantiate it. On Julia < 1.11 the [sources] table is ignored, so the in-repo
# path deps (this sublibrary and its in-repo siblings) are developed first.
function activate_group_env(dir)
    Pkg.activate(joinpath(@__DIR__, dir))
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop([
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "BracketingNonlinearSolve")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "NonlinearSolveBase")),
        ])
    end
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    include("core/exotic_type_tests.jl")
    include("core/forward_diff_tests.jl")
    include("core/least_squares_tests.jl")
    include("core/matrix_resizing_tests.jl")
    include("core/rootfind_tests.jl")
    # QA runs last: activate_group_env switches the active project to test/qa.
    activate_group_env("qa")
    @safetestset "Aqua" include("qa/qa.jl")
    @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
end

# Dep-adding groups run in their own isolated sub-envs (excluded from the
# base/Core env). SciMLSensitivity (Adjoint) and CUDA (gpu) are no longer
# Pkg.add'ed into the main resolve.
if GROUP == "Adjoint"
    activate_group_env("adjoint")
    include("adjoint/adjoint_tests.jl")
end

if GROUP == "CUDA"
    activate_group_env("gpu")
    include("gpu/cuda_tests.jl")
end
