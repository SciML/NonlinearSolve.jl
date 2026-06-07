using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

@info "Running tests for group: $(GROUP)"

if GROUP == "All" || GROUP == "Core"
    include("inference_tests.jl")
    include("least_squares_tests.jl")
    include("misc_tests.jl")
    include("qa_tests.jl")
    include("rootfind_tests.jl")
    include("sparsity_tests.jl")
end
