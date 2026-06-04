using TestItemRunner, InteractiveUtils, Pkg, Test

@info sprint(InteractiveUtils.versioninfo)

# The root NonlinearSolve runtests dispatcher activates this sublibrary and sets
# NLS_TEST_GROUP to the bare standard section name. Standard sublibrary groups:
#   Core — functional/correctness, incl. the folded adjoint test (SciMLSensitivity)
#   QA   — Aqua + Explicit Imports + the folded allocation test (AllocCheck)
#   GPU  — CUDA tests; the dedicated GPU.yml workflow sets GROUP="cuda".
const GROUP = get(ENV, "NLS_TEST_GROUP", "All")

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
