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

# Centralized SublibraryCI (sublibrary-tests.yml@v1) emits GROUP="<pkg>" for the
# Core section and GROUP="<pkg>_<Section>" for other sections. Decode it back to a
# bare section name so the dispatch below (and bare "all" for local runs) keeps
# working; the "MacOS" suffix only selects a runner, not a different selection.
# The Core section corresponds to the old bespoke CI's `core` group (the :core
# tag), not "all", so map it to "core".
const _G = get_from_test_args_or_env("GROUP", "all")
const _SUB = "SimpleNonlinearSolve"
const _SEC = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
const _SEC_BASE = endswith(_SEC, "MacOS") ? _SEC[1:(end - 5)] : _SEC
const GROUP = _SEC_BASE == "Core" ? "core" : lowercase(_SEC_BASE)

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
