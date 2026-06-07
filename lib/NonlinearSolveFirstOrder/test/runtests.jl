using SafeTestsets, Test, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = lowercase(get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "all")))

@info "Running tests for group: $(GROUP)"

if GROUP == "all" || GROUP == "core"
    include("inference_tests.jl")
    include("least_squares_tests.jl")
    include("misc_tests.jl")
    include("qa_tests.jl")
    include("rootfind_tests.jl")
    include("sparsity_tests.jl")
end
