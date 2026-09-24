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
        return include("basic_tests.jl")
    end,
    groups = Dict(
        # Wrappers runs in the base test env and is part of the "All" run.
        "Wrappers" => function ()
            return include("wrappers_tests.jl")
        end,
    ),
    # QA (Aqua/JET/ExplicitImports via SciMLTesting.run_qa) is a dep-adding group: it
    # runs in its own isolated sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl"),
    ),
)
