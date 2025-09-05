using TestItemRunner, InteractiveUtils, Pkg, Test

@info sprint(InteractiveUtils.versioninfo)

function parse_test_args()
    test_args_from_env = @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS
    test_args = Dict{String, String}()
    for arg in test_args_from_env
        if contains(arg, "=")
            key, value = split(arg, "="; limit = 2)
            test_args[key] = value
        end
    end
    @info "Parsed test args" test_args
    return test_args
end

const PARSED_TEST_ARGS = parse_test_args()

function get_from_test_args_or_env(key, default)
    haskey(PARSED_TEST_ARGS, key) && return PARSED_TEST_ARGS[key]
    return get(ENV, key, default)
end

const GROUP = lowercase(get_from_test_args_or_env("GROUP", "all"))

(GROUP == "all" || GROUP == "cuda") && Pkg.add(["CUDA"])
(GROUP == "all" || GROUP == "adjoint") && Pkg.add(["SciMLSensitivity"])
(GROUP == "all" || GROUP == "alloc_check") && Pkg.add(["AllocCheck"])

@testset "SimpleNonlinearSolve.jl" begin
    if GROUP == "all"
        @run_package_tests
    else
        @run_package_tests filter = ti -> (Symbol(GROUP) in ti.tags)
    end
end
