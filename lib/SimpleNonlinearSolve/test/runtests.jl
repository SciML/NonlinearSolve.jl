using SafeTestsets, Test, InteractiveUtils
using SciMLTesting

@info sprint(InteractiveUtils.versioninfo)

# SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP for local runs.
if !haskey(ENV, "NONLINEARSOLVE_TEST_GROUP") && haskey(ENV, "GROUP")
    ENV["NONLINEARSOLVE_TEST_GROUP"] = ENV["GROUP"]
end

run_tests(;
    env = "NONLINEARSOLVE_TEST_GROUP",
    core = function ()
        include("core/exotic_type_tests.jl")
        include("core/forward_diff_tests.jl")
        include("core/least_squares_tests.jl")
        include("core/matrix_resizing_tests.jl")
        return include("core/rootfind_tests.jl")
    end,
    groups = Dict(
        # Dep-adding groups run in their own isolated sub-envs (excluded from the
        # base/Core env and from the "All" run). Adjoint (SciMLSensitivity), Alloc
        # (AllocCheck) and CUDA carry deps beyond the base test set.
        "Adjoint" => (;
            env = joinpath(@__DIR__, "adjoint"),
            body = function ()
                return include("adjoint/adjoint_tests.jl")
            end,
        ),
        "Alloc" => (;
            env = joinpath(@__DIR__, "alloc"),
            body = function ()
                return include("alloc/allocation_tests.jl")
            end,
        ),
        "CUDA" => (;
            env = joinpath(@__DIR__, "gpu"),
            body = function ()
                return include("gpu/cuda_tests.jl")
            end,
        ),
    ),
    # QA (Aqua/ExplicitImports via SciMLTesting.run_qa) is a dep-adding group: it runs
    # in its own isolated sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl"),
    ),
    # "All" runs only the base-env Core group; the dep-adding groups (Adjoint, Alloc,
    # CUDA) and QA run only when selected by name.
    all = ["Core"],
)
