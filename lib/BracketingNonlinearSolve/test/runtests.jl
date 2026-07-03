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
        include("muller_tests.jl")
        return include("rootfind_tests.jl")
    end,
    groups = Dict(
        # Adjoint (Zygote) is a dep-adding group: it runs in its own isolated sub-env
        # under test/adjoint (excluded from the base/Core/All run) so Zygote's joint
        # graph does not constrain the base resolve.
        "Adjoint" => (;
            env = joinpath(@__DIR__, "adjoint"),
            body = function ()
                return include("adjoint/adjoint_tests.jl")
            end,
        ),
    ),
    # QA (Aqua/ExplicitImports via SciMLTesting.run_qa) is a dep-adding group: it runs
    # in its own isolated sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl"),
    ),
    # "All" runs only the base-env Core group; the dep-adding Adjoint group and QA
    # run only when selected by name.
    all = ["Core"],
)
