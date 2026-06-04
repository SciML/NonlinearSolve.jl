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
# Core section and GROUP="<pkg>_<Section>" for other sections; strip the prefix
# back to the bare standard section name. Standard sublibrary groups:
#   Core — functional/correctness, incl. the folded adjoint test (SciMLSensitivity)
#   QA   — Aqua + Explicit Imports + the folded allocation test (AllocCheck)
#   GPU  — CUDA tests; the dedicated GPU.yml workflow sets GROUP="cuda".
const _G = get_from_test_args_or_env("GROUP", "All")
const _SUB = "SimpleNonlinearSolve"
const GROUP = _G == _SUB ? "Core" :
    (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)

const _IS_ALL = GROUP in ("All", "all")
const _IS_CORE = GROUP in ("Core", "core")
const _IS_QA = GROUP in ("QA", "qa")
const _IS_GPU = GROUP in ("GPU", "gpu", "cuda")

(_IS_ALL || _IS_GPU) && Pkg.add(["CUDA"])
(_IS_ALL || _IS_CORE) && Pkg.add(["SciMLSensitivity"])
(_IS_ALL || _IS_QA) && Pkg.add(["AllocCheck"])

@testset "SimpleNonlinearSolve.jl" begin
    if _IS_ALL
        @run_package_tests
    elseif _IS_CORE
        @run_package_tests filter = ti -> (:core in ti.tags)
    elseif _IS_QA
        @run_package_tests filter = ti -> (:qa in ti.tags)
    elseif _IS_GPU
        @run_package_tests filter = ti -> (:cuda in ti.tags)
    else
        @run_package_tests filter = ti -> (Symbol(lowercase(GROUP)) in ti.tags)
    end
end
