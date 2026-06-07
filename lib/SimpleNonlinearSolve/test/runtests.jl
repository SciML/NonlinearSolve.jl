using SafeTestsets, Test, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

# Group dispatch: SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP.
const GROUP = get(ENV, "NONLINEARSOLVE_TEST_GROUP", get(ENV, "GROUP", "All"))

@info "Running tests for group: $(GROUP)"

# Heavy/optional group deps are added on demand (not part of the default
# resolve) so the Core/QA matrix stays lightweight, matching the original setup.
(GROUP == "All" || GROUP == "CUDA") && Pkg.add(["CUDA"])
(GROUP == "All" || GROUP == "Adjoint") && Pkg.add(["SciMLSensitivity"])

if GROUP == "All" || GROUP == "Adjoint"
    include("core/adjoint_tests.jl")
end

if GROUP == "All" || GROUP == "Core"
    include("core/exotic_type_tests.jl")
    include("core/forward_diff_tests.jl")
    include("core/least_squares_tests.jl")
    include("core/matrix_resizing_tests.jl")
    include("core/qa_tests.jl")
    include("core/rootfind_tests.jl")
end

if GROUP == "All" || GROUP == "CUDA"
    include("gpu/cuda_tests.jl")
end
