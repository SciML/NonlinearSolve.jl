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
        @safetestset "AllRoots" include("allroots.jl")
        return @safetestset "Single Root" include("single_root.jl")
    end,
    # QA (Aqua) is a dep-adding group: it runs in its own isolated sub-env under
    # test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = function ()
            return @safetestset "Code quality (Aqua.jl)" include("qa/qa.jl")
        end,
    ),
)
