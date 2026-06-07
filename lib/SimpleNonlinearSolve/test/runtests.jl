using SafeTestsets, Test, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = lowercase(get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "all")))

@info "Running tests for group: $(GROUP)"

# Heavy/optional group deps are added on demand (not part of the default
# resolve) so the Core/QA matrix stays lightweight, matching the original setup.
(GROUP == "all" || GROUP == "cuda") && Pkg.add(["CUDA"])
(GROUP == "all" || GROUP == "adjoint") && Pkg.add(["SciMLSensitivity"])

if GROUP == "all" || GROUP == "adjoint"
    include("core/adjoint_tests.jl")
end

if GROUP == "all" || GROUP == "core"
    include("core/exotic_type_tests.jl")
    include("core/forward_diff_tests.jl")
    include("core/least_squares_tests.jl")
    include("core/matrix_resizing_tests.jl")
    include("core/qa_tests.jl")
    include("core/rootfind_tests.jl")
end

if GROUP == "all" || GROUP == "cuda"
    include("gpu/cuda_tests.jl")
end
