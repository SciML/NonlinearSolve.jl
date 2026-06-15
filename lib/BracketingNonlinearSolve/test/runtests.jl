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
        # Adjoint runs in the base test env (Zygote is a base test dep) and is part
        # of the "All" run.
        "Adjoint" => function ()
            return include("adjoint_tests.jl")
        end,
    ),
    # QA (Aqua/ExplicitImports) is a dep-adding group: it runs in its own isolated
    # sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = function ()
            @safetestset "Aqua" include("qa/qa.jl")
            return @safetestset "Explicit Imports" include("qa/explicit_imports.jl")
        end,
    ),
    all = ["Core", "Adjoint"],
)
