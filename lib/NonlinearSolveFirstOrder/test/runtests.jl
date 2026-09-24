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
        include("conditioning_tests.jl")
        include("inference_tests.jl")
        include("least_squares_tests.jl")
        @safetestset "Bounded least-squares default" include("bounded_default_tests.jl")
        include("misc_tests.jl")
        @safetestset "Native bounded methods" include("native_bounded_tests.jl")
        include("rootfind_tests.jl")
        include("sparsity_tests.jl")
        return @safetestset "SciMLOperator Jacobians" include("operator_jacobian.jl")
    end,
    groups = Dict("NativeBounds" => (() -> @safetestset "Native bounded methods" include("native_bounded_tests.jl"))),
    all = ["Core"],
    # QA (Aqua/ExplicitImports via SciMLTesting.run_qa) is a dep-adding group: it runs
    # in its own isolated sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl"),
    ),
)
